<div align="center">

# Um Modelo Computacional Generativo para Design de Anticorpos

[![Homepage](http://img.shields.io/badge/Homepage-IgGM-ff88dd.svg)](https://iggm.rubo.wang)
[![Journal Paper](http://img.shields.io/badge/Journal_paper-biorxiv-FFA876.svg)](https://www.biorxiv.org/content/10.1101/2025.09.12.675771)
[![Conference Paper](http://img.shields.io/badge/Conference_paper-ICLR2025-6B77FE.svg)](https://openreview.net/forum?id=zmmfsJpYcq)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/TencentAI4S/IgGM/blob/master/LICENSE)


![header](docs/IgGM_dynamic.gif)

</div>


--------------------------------------------------------------------------------
[English](./README.md) | [简体中文](./README-zh.md) | [Español](./README-es.md) | Português

## 🔊Notícias

* **2025-08-22**: Acabamos de saber que nosso uso do IgGM na competição de design de anticorpos ([AIntibody: an experimentally validated in silico antibody discovery design challenge](https://www.nature.com/articles/s41587-024-02469-9)) nos rendeu um prêmio entre os três primeiros! 🎉
* **2025-08-21**: O IgGM foi atualizado para um modelo fundamental generativo para design de anticorpos, suportando tarefas como design de novos anticorpos, maturação de afinidade, design inverso, previsão de estrutura e humanização.
* **2025-01-15**: O IgGM foi aceito na ICLR 2025, com o artigo intitulado "IgGM: A Generative Model for Functional Antibody and Nanobody Design"🎉

## 📘Introdução

Este repositório contém a implementação dos dois artigos a seguir:

O artigo da ICLR 2025, "IgGM: A Generative Model for Functional Antibody and Nanobody Design," introduz o IgGM, um modelo que pode projetar a estrutura geral e sequências da região CDR com base em uma sequência de estrutura (‘framework’) fornecida, e também pode projetar anticorpos para epítopos específicos.

"A Generative Foundation Model for Antibody Design" estende ainda mais as capacidades do IgGM para um modelo fundamental generativo para design de anticorpos, permitindo tarefas como design de novo de anticorpos, maturação de afinidade, design inverso, previsão de estrutura e humanização.

Se você tiver alguma dúvida, entre em contato com a equipe do IgGM em wangrubo@hotmail.com, wufandi@outlook.com.

## 🧑🏻‍💻Começando

###
1. Clone o pacote
```shell
git clone https://github.com/TencentAI4S/IgGM.git
cd IgGM
```

2. Instale o ambiente

```shell
conda env create -n IgGM -f environment.yaml
conda activate IgGM
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
**Opcional:** 

Se você precisar usar relax para a saída, instale a seguinte versão do PyRosetta:

```shell
pip install https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.ubuntu.wheel/pyrosetta-2025.37+release.df75a9c48e-cp310-cp310-linux_x86_64.whl
```
Alternativamente, você pode usar OpenMM para relaxamento, que está incluído no environment.yaml. Use o argumento `--relax_open`.

3. Baixe o modelo (Opcional, os pesos pré-treinados serão baixados automaticamente quando o código for executado)
    * [Zenodo](https://zenodo.org/records/16909543)


**Nota**:

Se você baixar os pesos para a pasta `./checkpoints`, você poderá executar o código subsequente diretamente.

Se você não baixar os pesos, eles serão baixados automaticamente quando você executar o código.

## 📖Exemplos

Você pode usar um arquivo fasta como entrada de sequência e um arquivo pdb como entrada de antígeno. Arquivos de exemplo estão localizados na pasta `examples`.

* **Uma versão Colab do IgGM mantida por Luis pode ser encontrada em [Colab-IgGM](https://github.com/Lefrunila/Colab-IgGM), obrigado a Luis por suas contribuições!**

* **Opcional:**
  * Para todos os comandos, você pode usar PyRosetta para relaxar a saída adicionando `--relax` ou `-r`. Esta opção também adicionará átomos da cadeia lateral.
  * Alternativamente, use `--relax_open` ou `-r_open` para usar OpenMM para relaxamento de estrutura (sem necessidade de licença PyRosetta).
  * Para todos os comandos, você pode especificar o comprimento máximo de truncamento para o antígeno para 384 para evitar problemas de memória adicionando `--max_antigen_size 384` ou `-mas 384`.

Para o processamento subsequente, você precisa preparar um arquivo fasta e um arquivo pdb. Seu arquivo fasta deve ter a seguinte estrutura, que você pode consultar na pasta `examples`.

```
>H  # ID da cadeia pesada
VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS
>L # ID da cadeia leve
DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK
>A # ID do antígeno, precisa ser consistente com o arquivo pdb
NLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGP
```
* 'X' indica a região a ser projetada.
* Para obter o epítopo do antígeno, você pode usar o seguinte comando:
<a id="calculo-de-epitopo"></a>

```
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --cal_epitope

'--antigen' representa a estrutura de um complexo conhecido, e '--fasta' representa a sequência de um complexo conhecido, o que retornará o formato de epítopo necessário mais tarde, e após a cópia, o fasta pode ser substituído pela sequência que você precisa projetar. 

O formato de epítopo gerado é (O número de série começa em 1): 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 

Se você especificar o epítopo de acordo com a sequência, certifique-se de que a ordem da sequência seja consistente com a ordem no arquivo PDB e marque o número de série da posição correspondente.
```

#### Exemplo 1: Usando IgGM para prever estruturas de anticorpos e nanoanticorpos
* Se o PDB contiver a estrutura do complexo, este comando gerará automaticamente informações de epítopo. Neste caso, você pode remover `--epitope`.
```
# anticorpo
python design.py --fasta examples/fasta.files.native/8iv5_A_B_G.fasta --antigen examples/pdb.files.native/8iv5_A_B_G.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanoanticorpo
python design.py --fasta examples/fasta.files.native/8q94_C_NA_A.fasta --antigen examples/pdb.files.native/8q94_C_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```

#### Exemplo 2: Dada a estrutura de um complexo, usar IgGM para projetar a sequência correspondente
* Se o PDB contiver a estrutura do complexo, este comando gerará automaticamente informações de epítopo. Neste caso, você pode remover `--epitope`.
```
# anticorpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164 --run_task inverse_design

# nanoanticorpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_H3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167 --run_task inverse_design
```

#### Exemplo 3: Usando IgGM para redesenho de sequência de região de estrutura (framework)
* Aqui tomamos a humanização como exemplo, o que requer [BioPhi](https://biophi.dichlab.org/humanization/humanize/).
```
# Anticorpo de camundongo inicial
>H
QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS
>L
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Apenas para humanização, use BioPhi para a humanização inicial do anticorpo de camundongo
>H
QVQLQESGPGLVKPSETLSLTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Compare as diferenças entre as sequências humanizadas de diferentes regiões FR e o anticorpo de camundongo para identificar as partes que precisam de otimização. Aqui, FR1 é usado como exemplo.
>H
QVQLQESGPGLVXPSXXLSXTCTVSGFSLTGYGWGWIRQPPGKGLEWIGSIWGDGNTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARERDYRLDYWGQGTLVTVSS
>L
DIQLTQSPSFLSASVGDRVTITCRASGNIHNYLAWYQQKPGKAPKLLIYYTTTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHFWSTPRTFGGGTKVEIK
>A
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

# Use IgGM para projetar a região FR1

python design.py --fasta examples/humanization/fasta.files.design.heavy_fr1/1vfb_B_A_C.fasta --antigen examples/humanization/pdb.files.native/1vfb_B_A_C.pdb --run_task fr_design

```

#### Exemplo 4: Usando IgGM para maturação de afinidade de anticorpos e nanoanticorpos contra um determinado antígeno.
```
# anticorpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --fasta_origin examples/fasta.files.native/8hpu_M_N_A.fasta --run_task affinity_maturation --num_samples 100

# nanoanticorpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --fasta_origin examples/fasta.files.native/8q95_B_NA_A.fasta --run_task affinity_maturation --num_samples 100

# Se você tiver várias GPUs, pode executar o seguinte comando para executar em paralelo
bash scripts/multi_runs.sh

# Para as sequências geradas, você pode consultar o seguinte arquivo para coletar e salvar os resultados, bem como visualizá-los
scripts/Merge_output.ipynb

# Executar o comando acima gerará uma pasta nomeada com o ID de design correspondente, que contém o seguinte:
- outputs/maturation/results/8hpu_M_N_A
- outputs/maturation/results/8hpu_M_N_A/dup # Gráfico de distribuição de aminoácidos para sequências duplicadas
- outputs/maturation/results/8hpu_M_N_A/dup/logo.png
- outputs/maturation/results/8hpu_M_N_A/dup/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/original # Gráfico de distribuição de aminoácidos para a sequência original
- outputs/maturation/results/8hpu_M_N_A/original/logo.png
- outputs/maturation/results/8hpu_M_N_A/original/stacked_bar_chart.png
- outputs/maturation/results/8hpu_M_N_A/dedup.csv # Resultados estatísticos para sequências desduplicadas
- outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv # Resultados estatísticos para sequências desduplicadas (incluindo frequência de geração)
- outputs/maturation/results/8hpu_M_N_A/dup.csv # Resultados estatísticos para sequências duplicadas

# Filtrar com base na frequência usando outputs/maturation/results/8hpu_M_N_A/dedup_diff_freq.csv

```

#### Exemplo 5: Usando IgGM para projetar a sequência do loop CDR H3 para um anticorpo e nanoanticorpo contra um determinado antígeno, e prever a estrutura geral.
```
# anticorpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanoanticorpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/C8q95_B_NA_A_DR_3.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb

```

#### Exemplo 6: Usando IgGM para projetar as sequências do loop CDR para um anticorpo e nanoanticorpo contra um determinado antígeno e prever a estrutura geral.
```
# anticorpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb

# nanoanticorpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb
```

Você pode especificar outras regiões para design; explore mais exemplos na pasta examples.

#### Exemplo 7: Projetar sequências de loop CDR de anticorpos e nanoanticorpos e prever a estrutura geral com base apenas em um determinado antígeno e epítopo de ligação, sem a necessidade da estrutura do complexo.
* **É possível projetar um anticorpo para um epítopo completamente novo.**
```
# anticorpo
python design.py --fasta examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_All.fasta --antigen examples/pdb.files.native/8hpu_M_N_A.pdb --epitope 7 8 9 10 11 12 13 14 108 109 110 111 112 113 114 115 116 118 167 157 158 160 161 162 163 164

# nanoanticorpo
python design.py --fasta examples/fasta.files.design/8q95_B_NA_A/8q95_B_NA_A_CDR_All.fasta --antigen examples/pdb.files.native/8q95_B_NA_A.pdb --epitope 109 110 111 112 113 114 115 116 117 120 121 140 141 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 165 166 167
```
Para um antígeno completamente novo, você pode especificar epítopos para projetar anticorpos que podem se ligar a eles.

#### Exemplo 8: Mesclar várias cadeias de antígenos em uma única cadeia, especificando os IDs das cadeias a serem mescladas.

```
python scripts/merge_chains.py --antigen examples/pdb.files.native/8ucd.pdb --output ./outputs --merge_ids A_B_C
```

#### Exemplo 9: Recortar um antígeno para economizar memória antes da inferência.
* **Importante!!** O epítopo deve ser calculado primeiro (veja [Cálculo de Epítopo](#calculo-de-epitopo)). Se houver várias cadeias de antígenos, mescle primeiro (Exemplo 8).
* As cadeias de anticorpos (H/L) são preservadas inalteradas; apenas a cadeia de antígeno especificada é recortada.
```bash
python scripts/trim_antigen.py --pdb outputs/8ucd_merge.pdb --fasta outputs/8ucd_merge.fasta --output outputs/8ucd_merge_trimmed.pdb --antigen-chain A --epitope 198 199 200 201 202 203 204 --keep-radius 10.0
```
* **--keep-radius**: (Padrão 10.0Å) Inclui resíduos dentro de X Ångstroms do centro de massa do epítopo, garantindo que o contexto estrutural seja preservado mesmo se os resíduos forem descontínuos na sequência. Defina como 0 para desabilitar.
* Esta ferramenta gera PDB e FASTA recortados e imprime os novos índices de epítopo renumerados para uso no design.

#### Exemplo 10: Executar inferência com um antígeno recortado e restaurar para o original.
* Este é o fluxo de trabalho recomendado para antígenos grandes: projetar contra um fragmento recortado, depois restaurar automaticamente o contexto completo do antígeno e relaxar o complexo final.
```bash
python design.py \
    --fasta outputs/8ucd_merge_trimmed.fasta \
    --antigen outputs/8ucd_merge_trimmed_noAb.pdb \
    --epitope 6 7 8 9 ... (índices da saída de corte anterior) \
    --restore_merged outputs/8ucd_merge.pdb \
    --restore_unmerged examples/pdb.files.native/8ucd.pdb \
    --restore_IDs A_B_C \
    --relax_open
```
* **--restore_merged**: O PDB usado para o recorte (fornece o quadro de referência).
* **--restore_unmerged**: O PDB completo original (fornece as cadeias exatas para saída).
* **--restore_IDs**: Os IDs da cadeia do PDB original para incluir na saída final.
* **--relax_open**: Realiza relaxamento OpenMM *após* a restauração, garantindo que a interface anticorpo-antígeno seja minimizada energeticamente no contexto do antígeno completo.

# 🤝🏻Licença

Nosso modelo e código são lançados sob a Licença MIT e podem ser usados livremente para fins acadêmicos e comerciais.

Se você tiver alguma dúvida, entre em contato com a equipe do IgGM em wangrubo@hotmail.com, wufandi@outlook.com.

## 📋️Citar IgGM

Se você usar IgGM em sua pesquisa, por favor cite nosso trabalho.

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
