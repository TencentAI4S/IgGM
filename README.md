<div align="center">

# A Generative Foundation Model for Antibody Design

[![Homepage](http://img.shields.io/badge/Homepage-IgGM-ff88dd.svg)](https://iggm.rubo.wang)
[![Journal Paper](http://img.shields.io/badge/Journal_paper-biorxiv-FFA876.svg)](https://www.biorxiv.org/content/10.1101/2025.09.12.675771)
[![Conference Paper](http://img.shields.io/badge/Conference_paper-ICLR2025-6B77FE.svg)](https://openreview.net/forum?id=zmmfsJpYcq)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/TencentAI4S/IgGM/blob/master/LICENSE)


![header](docs/IgGM_dynamic.gif)

</div>


--------------------------------------------------------------------------------
English | [简体中文](./README-zh.md)
## 🔊News

* **2025-08-22**: We just learned that our use of IgGM in the antibody design competition ([AIntibody: an experimentally validated in silico antibody discovery design challenge](https://www.nature.com/articles/s41587-024-02469-9)) has won us a top-three prize! 🎉
* **2025-08-21**: IgGM is updated to a generate foundation model for antibody design, supporting tasks such as novel antibody design, affinity maturation, reverse design, structure prediction, and humanization.
* **2025-01-15**: IgGM is accepted at ICLR 2025, with the paper titled "IgGM: A Generative Model for Functional Antibody and Nanobody Design"🎉

## 📘Introduction

This repository contains the implementation for the following two papers:

The ICLR 2025 paper, "IgGM: A Generative Model for Functional Antibody and Nanobody Design," introduces IgGM, a model that can design the overall structure and CDR region sequences based on a given framework sequence, and can also design antibodies for specific epitopes.

"A Generative Foundation Model for Antibody Design" further extends IgGM's capabilities to a generative foundation model for antibody design, enabling tasks such as de novo antibody design, affinity maturation, inverse design, structure prediction, and humanization.



If you have any questions, please contact the IgGM team at wangrubo@hotmail.com, wufandi@outlook.com.




## 🧑🏻‍💻Getting Started

###
1. Clone the package
```shell
git clone https://github.com/TencentAI4S/IgGM.git
cd IgGM
```

2. Install the environment

```shell
conda env create -n IgGM -f environment.yaml
conda activate IgGM
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
**Optional:** 

If you need to use relax for the output, please install the following version of PyRosetta:

```shell
pip install https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.ubuntu.wheel/pyrosetta-2025.37+release.df75a9c48e-cp310-cp310-linux_x86_64.whl
```

3. Download the model (Optional, pretrained weights will be downloaded automatically when the code is run)
    * [Zenodo](https://zenodo.org/records/16909543)


**Note**:

If you download the weights into the `./checkpoints` folder, you can run the subsequent code directly.

If you do not download the weights, they will be downloaded automatically when you run the code.

## 📖Examples

You can use a fasta file as the sequence input and a pdb file as the antigen input. Example files are located in the `examples` folder.

* **A Colab version of IgGM maintained by Luis can be found at [Colab-IgGM](https://github.com/Lefrunila/Colab-IgGM), thanks to Luis for his contributions!**

* **Optional:**
  * For all commands, you can use PyRosetta to relax the output by adding `--relax` or `-r`. This option will also add side-chain atoms.
  * For all commands, you can specify the maximum truncation length for the antigen to 384 to avoid memory issues by adding `--max_antigen_size 384` or `-mas 384`.

For subsequent processing, you need to prepare a fasta file and a pdb file. Your fasta file should have the following structure, which you can refer to in the `examples` folder.

```
>H  # Heavy chain ID
VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
>L # Light chain ID
DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
>A # Antigen ID, needs to be consistent with the pdb file
NLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGP
```
* 'X' indicates the region to be designed.
* To obtain the epitope of the antigen, you can use the following command:

```
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --cal_epitope

'--antigen' represents the structure of a known complex, and '--fasta' represents the sequence of a known complex, which will return the epitope format required later, and after copying, the fasta can be replaced with the sequence you need to design. 

The generated epitope format is (The serial number starts at 1): 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 

If you specify epitope according to the sequence, make sure that the order of the sequence is consistent with the order in the PDB file, and mark the serial number of the corresponding position.
```

#### Example 1: Using IgGM to predict antibody and nanobody structures
* If the PDB contains the structure of the complex, this command will automatically generate epitope information. In this case, you can remove `--epitope`.
```
# antibody
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanobody
python design.py --fasta examples/fasta.files.native/8q94_C_NA_A.fasta --antigen examples/pdb.files.native/8q94_C_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```

#### Example 2: Given the structure of a complex, use IgGM to design the corresponding sequence
* If the PDB contains the structure of the complex, this command will automatically generate epitope information. In this case, you can remove `--epitope`.
```
# antibody
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164 --run_task inverse_design

# nanobody
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_H3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167 --run_task inverse_design
```

#### Example 3: Using IgGM for framework region sequence redesign
* Here, we take humanization as an example, which requires [BioPhi](https://biophi.dichlab.org/humanization/humanize/).
```
# Initial mouse antibody
>H
QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS
>L
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# For humanization only, use BioPhi for initial humanization of the mouse antibody
>H
QVQLQESGPGLVKPSETLSLTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Compare the differences between the humanized sequences of different FR regions and the mouse antibody to identify the parts that need optimization. Here, FR1 is used as an example.
>H
QVQLQESGPGLVXPSXXLSXTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Use IgGM to design the FR1 region

python design.py --fasta examples/humanization/fasta.files.design.heavy_fr1/1vfb_B_A_C.fasta --antigen examples/humanization/pdb.files.native/1vfb_B_A_C.pdb --run_task fr_design

```

#### Example 4: Using IgGM for affinity maturation of antibodies and nanobodies against a given antigen.
```
# antibody
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --fasta_origin examples/fasta.files.native/8hpu_M_N_A.fasta --run_task affinity_maturation --num_samples 100

# nanobody
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --fasta_origin examples/fasta.files.native/8q95_B_NA_A.fasta --run_task affinity_maturation --num_samples 100

# If you have multiple GPUs, you can run the following command to execute in parallel
bash scripts/multi_runs.sh

# For the generated sequences, you can refer to the following file to collect and save the results, as well as visualize them
scripts/Merge_output.ipynb

# Executing the above command will generate a folder named after the corresponding design ID, which contains the following:
- outputs/maturation/results/8hpu_M_N_A
- outputs/maturation/results/8hpu_M_N_A/dup # Amino acid distribution plot for duplicated sequences
- outputs/maturation/results/8hpu_M_N_A/dup/logo.png
- outputs/maturation/results/8hpu_M_N_A/dup/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/original # Amino acid distribution plot for the original sequence
- outputs/maturation/results/8hpu_M_N_A/original/logo.png
- outputs/maturation/results/8hpu_M_N_A/original/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/dedup.csv # Statistical results for deduplicated sequences
- outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv # Statistical results for deduplicated sequences (including generation frequency)
- outputs/maturation/results/8hpu_M_N_A/dup.csv # Statistical results for duplicated sequences

# Filter based on frequency using outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv

```

#### Example 5: Using IgGM to design the CDR H3 loop sequence for an antibody and nanobody against a given antigen, and predict the overall structure.
```
# antibody
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanobody
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/C8q95_B_NA_A_DR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb

```

#### Example 6: Using IgGM to design the CDR loop sequences for an antibody and nanobody against a given antigen, and predict the overall structure.
```
# antibody
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanobody
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb
```

You can specify other regions for design; explore more examples in the examples folder.

#### Example 7: Design antibody and nanobody CDR loop sequences and predict the overall structure based only on a given antigen and binding epitope, without needing the complex structure.
* **It is possible to design an antibody for a completely new epitope.**
```
# antibody
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanobody
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```
For a completely new antigen, you can specify epitopes to design antibodies that can bind to them.

#### Example 8: Merge multiple antigen chains into a single chain, specifying the IDs of the chains to be merged.

```
python scripts/merge_chains.py --antigen examples/pdb.files.native/8ucd.pdb --output ./outputs --merge_ids A_B_C
```

# 🤝🏻License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.

If you have any questions, please contact the IgGM team at wangrubo@hotmail.com, wufandi@outlook.com.

## 📋️Citing IgGM

If you use IgGM in your research, please cite our work.

```BibTeX
@inproceedings{
wang2025iggm,
title={Ig{GM}: A Generative Model for Functional Antibody and Nanobody Design},
author={Wang, Rubo and Wu, Fandi and Gao, Xingyu and Wu, Jiaxiang and Zhao, Peilin and Yao, Jianhua},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=zmmfsJpYcq}
}
```
```BibTeX
@article {Wang2025.09.12.675771,
	author = {Wang, Rubo and Wu, Fandi and Shi, Jiale and Song, Yidong and Kong, Yu and Ma, Jian and He, Bing and Yan, Qihong and Ying, Tianlei and Zhao, Peilin and Gao, Xingyu and Yao, Jianhua},
	title = {A Generative Foundation Model for Antibody Design},
	year = {2025},
	doi = {10.1101/2025.09.12.675771},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/09/16/2025.09.12.675771},
	journal = {bioRxiv}
}
```
