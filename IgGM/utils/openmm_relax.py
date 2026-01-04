import os
import sys
import tempfile

def OpenMM_relax(pdb_file):
    """
    Relaxes a PDB structure using OpenMM and PDBFixer.
    Replaces the proprietary PyRosetta relaxation.
    
    Strategy:
    1. Fix residues/atoms with PDBFixer.
    2. Remove Heterogens (Ligands/Water) to ensure Amber14 compatibility.
    3. Save to temp PDB and STRIP CONECTs to clean topology.
    4. Reload, Add Hydrogens (Modeller), Minimize.
    """
    print(f"🧬 OpenMM processing {pdb_file} for Relax")
    
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, HBonds, NoCutoff
        from openmm import LangevinIntegrator, unit, Platform
    except ImportError:
        print("❌ OpenMM or PDBFixer not found. Please install them via conda: `conda install -c conda-forge openmm pdbfixer`")
        return

    # 1. Load and Fix PDB
    try:
        print("   Initializing PDBFixer...")
        fixer = PDBFixer(filename=pdb_file)
        
        print("   Removing heterogens (ligands/water)...")
        fixer.removeHeterogens(keepWater=False)
        
        print("   Fixing topology (missing residues/atoms)...")
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        # skip fixer.addMissingHydrogens - we use Modeller later
    except Exception as e:
        print(f"   ❌ Error validating/fixing PDB: {e}")
        return

    # 2. Save Intermediate and Strip CONECTs
    print("   Stripping bad CONECT records...")
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(temp_fd)
    
    try:
        # Write fixed PDB to temp
        with open(temp_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
        # Read back and strip CONECT lines
        with open(temp_path, 'r') as f:
            lines = f.readlines()
            
        clean_lines = [line for line in lines if not line.startswith("CONECT")]
        
        # Write clean PDB (reusing temp path)
        with open(temp_path, 'w') as f:
            f.writelines(clean_lines)
            
        # 3. Reload Clean PDB
        print("   Reloading clean topology...")
        pdb = PDBFile(temp_path)
        
    except Exception as e:
        print(f"   ❌ Error in Strip-CONECT step: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 4. Setup Modeller and ForceField
    try:
        # Standard choice: amber14-all.xml and implicit solvent like implicit/gbn2.xml
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # 5. Add Hydrogens using ForceField
        print("   Adding hydrogens...")
        modeller.addHydrogens(forcefield)
        
        # 6. Create System
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, 
                                         constraints=HBonds)
        
        # Integrator
        integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
        
        # Platform
        try:
            platform = Platform.getPlatformByName('CUDA')
            prop = {'CudaPrecision': 'mixed'}
        except Exception:
            try:
                platform = Platform.getPlatformByName('OpenCL')
                prop = {}
            except:
                platform = Platform.getPlatformByName('CPU')
                prop = {}
            
        print(f"   Using platform: {platform.getName()}")

        simulation = Simulation(modeller.topology, system, integrator, platform, prop)
        simulation.context.setPositions(modeller.positions)

        # 7. Minimize Energy
        print("   Minimizing energy...")
        simulation.minimizeEnergy()
        
        # 8. Save Relaxed PDB
        state = simulation.context.getState(getPositions=True)
        with open(pdb_file, 'w') as f:
            PDBFile.writeFile(simulation.topology, state.getPositions(), f)
            
        print(f"✅ OpenMM relaxed PDB saved to: {pdb_file}")
        
    except Exception as e:
        print(f"   ❌ OpenMM Relax failed: {e}")
