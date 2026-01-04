<div align="center">

# Un Modelo Fundacional Generativo para el Diseño de Anticuerpos

[![Homepage](http://img.shields.io/badge/Homepage-IgGM-ff88dd.svg)](https://iggm.rubo.wang)
[![Journal Paper](http://img.shields.io/badge/Journal_paper-biorxiv-FFA876.svg)](https://www.biorxiv.org/content/10.1101/2025.09.12.675771)
[![Conference Paper](http://img.shields.io/badge/Conference_paper-ICLR2025-6B77FE.svg)](https://openreview.net/forum?id=zmmfsJpYcq)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/TencentAI4S/IgGM/blob/master/LICENSE)


![header](docs/IgGM_dynamic.gif)

</div>


--------------------------------------------------------------------------------
[English](./README.md) | [简体中文](./README-zh.md) | Español | [Português](./README-pt.md)

## 🔊Noticias

* **2025-08-22**: ¡Acabamos de enterarnos de que nuestro uso de IgGM en la competencia de diseño de anticuerpos ([AIntibody: an experimentally validated in silico antibody discovery design challenge](https://www.nature.com/articles/s41587-024-02469-9)) nos ha ganado un premio entre los tres primeros! 🎉
* **2025-08-21**: IgGM se actualiza a un modelo fundacional generativo para el diseño de anticuerpos, admitiendo tareas como el diseño de nuevos anticuerpos, maduración de afinidad, diseño inverso, predicción de estructuras y humanización.
* **2025-01-15**: IgGM es aceptado en ICLR 2025, con el artículo titulado "IgGM: A Generative Model for Functional Antibody and Nanobody Design"🎉

## 📘Introducción

Este repositorio contiene la implementación de los dos artículos siguientes:

El artículo de ICLR 2025, "IgGM: A Generative Model for Functional Antibody and Nanobody Design," introduce IgGM, un modelo que puede diseñar la estructura general y las secuencias de la región CDR basándose en una secuencia marco dada, y también puede diseñar anticuerpos para epítopos específicos.

"A Generative Foundation Model for Antibody Design" extiende aún más las capacidades de IgGM a un modelo fundacional generativo para el diseño de anticuerpos, permitiendo tareas como el diseño de novo de anticuerpos, maduración de afinidad, diseño inverso, predicción de estructuras y humanización.

Si tiene alguna pregunta, comuníquese con el equipo de IgGM en wangrubo@hotmail.com, wufandi@outlook.com.

## 🧑🏻‍💻Comenzando

###
1. Clonar el paquete
```shell
git clone https://github.com/TencentAI4S/IgGM.git
cd IgGM
```

2. Instalar el entorno

```shell
conda env create -n IgGM -f environment.yaml
conda activate IgGM
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
**Opcional:** 

Si necesita usar relax para la salida, instale la siguiente versión de PyRosetta:

```shell
pip install https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.ubuntu.wheel/pyrosetta-2025.37+release.df75a9c48e-cp310-cp310-linux_x86_64.whl
```
Alternativamente, puede usar OpenMM para la relajación, que está incluido en environment.yaml. Use el argumento `--relax_open`.

3. Descargar el modelo (Opcional, los pesos preentrenados se descargarán automáticamente cuando se ejecute el código)
    * [Zenodo](https://zenodo.org/records/16909543)


**Nota**:

Si descarga los pesos en la carpeta `./checkpoints`, puede ejecutar el código subsecuente directamente.

Si no descarga los pesos, se descargarán automáticamente cuando ejecute el código.

## 📖Ejemplos

Puede usar un archivo fasta como entrada de secuencia y un archivo pdb como entrada de antígeno. Los archivos de ejemplo se encuentran en la carpeta `examples`.

* **Una versión de Colab de IgGM mantenida por Luis se puede encontrar en [Colab-IgGM](https://github.com/Lefrunila/Colab-IgGM), ¡gracias a Luis por sus contribuciones!**

* **Opcional:**
  * Para todos los comandos, puede usar PyRosetta para relajar la salida agregando `--relax` o `-r`. Esta opción también agregará átomos de cadena lateral.
  * Alternativamente, use `--relax_open` o `-r_open` para usar OpenMM para la relajación de estructuras (no se requiere licencia de PyRosetta).
  * Para todos los comandos, puede especificar la longitud máxima de truncamiento para el antígeno en 384 para evitar problemas de memoria agregando `--max_antigen_size 384` o `-mas 384`.

Para el procesamiento posterior, necesita preparar un archivo fasta y un archivo pdb. Su archivo fasta debe tener la siguiente estructura, la cual puede consultar en la carpeta `examples`.

```
>H  # ID de cadena pesada
VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
>L # ID de cadena ligera
DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
>A # ID del antígeno, debe ser consistente con el archivo pdb
NLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGP
```
* 'X' indica la región a diseñar.
* Para obtener el epítopo del antígeno, puede usar el siguiente comando:
<a id="calculo-de-epitopo"></a>

```
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --cal_epitope

'--antigen' representa la estructura de un complejo conocido, y '--fasta' representa la secuencia de un complejo conocido, lo cual devolverá el formato de epítopo requerido más tarde, y después de copiar, el fasta puede ser reemplazado con la secuencia que necesita diseñar. 

El formato de epítopo generado es (El número de serie comienza en 1): 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 

Si especifica el epítopo según la secuencia, asegúrese de que el orden de la secuencia sea consistente con el orden en el archivo PDB, y marque el número de serie de la posición correspondiente.
```

#### Ejemplo 1: Usando IgGM para predecir estructuras de anticuerpos y nanoanticuerpos
* Si el PDB contiene la estructura del complejo, este comando generará automáticamente la información del epítopo. En este caso, puede eliminar `--epitope`.
```
# anticuerpo
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanoanticuerpo
python design.py --fasta examples/fasta.files.native/8q94_C_NA_A.fasta --antigen examples/pdb.files.native/8q94_C_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```

#### Ejemplo 2: Dada la estructura de un complejo, usar IgGM para diseñar la secuencia correspondiente
* Si el PDB contiene la estructura del complejo, este comando generará automáticamente la información del epítopo. En este caso, puede eliminar `--epitope`.
```
# anticuerpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164 --run_task inverse_design

# nanoanticuerpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_H3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167 --run_task inverse_design
```

#### Ejemplo 3: Usando IgGM para el rediseno de secuencias de la región marco (framework)
* Aquí tomamos la humanización como ejemplo, que requiere [BioPhi](https://biophi.dichlab.org/humanization/humanize/).
```
# Anticuerpo de ratón inicial
>H
QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS
>L
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Solo para humanización, use BioPhi para la humanización inicial del anticuerpo de ratón
>H
QVQLQESGPGLVKPSETLSLTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Compare las diferencias entre las secuencias humanizadas de diferentes regiones FR y el anticuerpo de ratón para identificar las partes que necesitan optimización. Aquí, se usa FR1 como ejemplo.
>H
QVQLQESGPGLVXPSXXLSXTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Use IgGM para diseñar la región FR1

python design.py --fasta examples/humanization/fasta.files.design.heavy_fr1/1vfb_B_A_C.fasta --antigen examples/humanization/pdb.files.native/1vfb_B_A_C.pdb --run_task fr_design

```

#### Ejemplo 4: Usando IgGM para la maduración de afinidad de anticuerpos y nanoanticuerpos contra un antígeno dado.
```
# anticuerpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --fasta_origin examples/fasta.files.native/8hpu_M_N_A.fasta --run_task affinity_maturation --num_samples 100

# nanoanticuerpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --fasta_origin examples/fasta.files.native/8q95_B_NA_A.fasta --run_task affinity_maturation --num_samples 100

# Si tiene múltiples GPUs, puede ejecutar el siguiente comando para ejecutar en paralelo
bash scripts/multi_runs.sh

# Para las secuencias generadas, puede referirse al siguiente archivo para recolectar y guardar los resultados, así como visualizarlos
scripts/Merge_output.ipynb

# Ejecutar el comando anterior generará una carpeta llamada con el ID de diseño correspondiente, que contiene lo siguiente:
- outputs/maturation/results/8hpu_M_N_A
- outputs/maturation/results/8hpu_M_N_A/dup # Gráfico de distribución de aminoácidos para secuencias duplicadas
- outputs/maturation/results/8hpu_M_N_A/dup/logo.png
- outputs/maturation/results/8hpu_M_N_A/dup/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/original # Gráfico de distribución de aminoácidos para la secuencia original
- outputs/maturation/results/8hpu_M_N_A/original/logo.png
- outputs/maturation/results/8hpu_M_N_A/original/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/dedup.csv # Resultados estadísticos para secuencias deduplicadas
- outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv # Resultados estadísticos para secuencias deduplicadas (incluyendo frecuencia de generación)
- outputs/maturation/results/8hpu_M_N_A/dup.csv # Resultados estadísticos para secuencias duplicadas

# Filtrar basado en frecuencia usando outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv

```

#### Ejemplo 5: Usando IgGM para diseñar la secuencia del bucle CDR H3 para un anticuerpo y nanoanticuerpo contra un antígeno dado, y predecir la estructura general.
```
# anticuerpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanoanticuerpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/C8q95_B_NA_A_DR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb

```

#### Ejemplo 6: Usando IgGM para diseñar las secuencias de bucle CDR para un anticuerpo y nanoanticuerpo contra un antígeno dado, y predecir la estructura general.
```
# anticuerpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanoanticuerpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb
```

Puede especificar otras regiones para el diseño; explore más ejemplos en la carpeta examples.

#### Ejemplo 7: Diseñar secuencias de bucle CDR de anticuerpos y nanoanticuerpos y predecir la estructura general basándose solo en un antígeno y epítopo de unión dados, sin necesitar la estructura del complejo.
* **Es posible diseñar un anticuerpo para un epítopo completamente nuevo.**
```
# anticuerpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanoanticuerpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```
Para un antígeno completamente nuevo, puede especificar epítopos para diseñar anticuerpos que puedan unirse a ellos.

#### Ejemplo 8: Fusionar múltiples cadenas de antígenos en una sola cadena, especificando los IDs de las cadenas a fusionar.

```
python scripts/merge_chains.py --antigen examples/pdb.files.native/8ucd.pdb --output ./outputs --merge_ids A_B_C
```

#### Ejemplo 9: Recortar un antígeno para ahorrar memoria antes de la inferencia.
* **¡Importante!** El epítopo debe calcularse primero (ver [Cálculo de Epítopo](#calculo-de-epitopo)). Si hay múltiples cadenas de antígenos, fusione primero (Ejemplo 8).
* Las cadenas de anticuerpos (H/L) se conservan sin cambios; solo la cadena de antígeno especificada se recorta.
```bash
python scripts/trim_antigen.py --pdb outputs/8ucd_merge.pdb --fasta outputs/8ucd_merge.fasta --output outputs/8ucd_merge_trimmed.pdb --antigen-chain A --epitope 198 199 200 201 202 203 204 --keep-radius 10.0
```
* **--keep-radius**: (Predeterminado 10.0Å) Incluye residuos dentro de X Ångstroms del centro de masa del epítopo, asegurando que el contexto estructural se preserve incluso si los residuos son discontinuos en la secuencia. Establecer en 0 para deshabilitar.
* Esta herramienta genera tanto PDB como FASTA recortados, e imprime los nuevos índices de epítopo renumerados para su uso en el diseño.

#### Ejemplo 10: Ejecutar inferencia con un antígeno recortado y restaurar al original.
* Este es el flujo de trabajo recomendado para antígenos grandes: Diseñar contra un parche recortado, luego restaurar automáticamente el contexto completo del antígeno y relajar el complejo final.
```bash
python design.py \
    --fasta outputs/8ucd_merge_trimmed.fasta \
    --antigen outputs/8ucd_merge_trimmed_noAb.pdb \
    --epitope 6 7 8 9 ... (índices de la salida de recorte) \
    --restore_merged outputs/8ucd_merge.pdb \
    --restore_unmerged examples/pdb.files.native/8ucd.pdb \
    --restore_IDs A_B_C \
    --relax_open
```
* **--restore_merged**: El PDB usado para el recorte (proporciona el marco de referencia).
* **--restore_unmerged**: El PDB completo original (proporciona las cadenas exactas para generar).
* **--restore_IDs**: Los IDs de cadena del PDB original para incluir en la salida final.
* **--relax_open**: Realiza relajación OpenMM *después* de la restauración, asegurando que la interfaz anticuerpo-antígeno se minimice energéticamente en el contexto del antígeno completo.

# 🤝🏻Licencia

Nuestro modelo y código se publican bajo la Licencia MIT, y pueden usarse libremente tanto para fines académicos como comerciales.

Si tiene alguna pregunta, comuníquese con el equipo de IgGM en wangrubo@hotmail.com, wufandi@outlook.com.

## 📋️Citar IgGM

Si usa IgGM en su investigación, por favor cite nuestro trabajo.

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
