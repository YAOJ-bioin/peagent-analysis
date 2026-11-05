# PEAgent

**Sequence-to-single-cell modelling of plant chromatin accessibility and its regulatory grammar across three crop species.**

PEAgent trains a per-species convolutional neural network that predicts the
single-cell chromatin-accessibility state of every accessible chromatin region
(ACR) directly from 1,344 bp of DNA sequence, and uses the trained models to
read out the regulatory grammar of plant genomes: a cross-species lexicon of
transcription-factor contribution patterns, their spatial co-occurrence and
synergy, cross-species transferability, ACR evolutionary constraint, and the
effect of natural sequence variants on accessibility.

This repository contains the analysis code accompanying the manuscript. Scripts
are organised as self-contained prototypes that document the method for each
figure; they read a small set of per-species inputs (atlas, model, peaks,
genome) and are pointed at a local data directory through `$PEAGENT_DATA_ROOT`.

## Species

| Species | Assembly | Nuclei | ACRs | Tissues | Chromosomes |
|---------|----------|--------|------|---------|-------------|
| Soybean (*Glycine max*) | Wm82.a4.v1 | 200,732 | 303,113 | 10 | `Gm1..Gm20` |
| Maize (*Zea mays*) | B73 NAM-5.0 | 50,639 | 158,586 | 6 | `chr1..chr10` |
| Rice (*Oryza sativa*) | Nipponbare MSU7 | 104,029 | 128,764 | 9 | `Chr1..Chr12` |

## Repository layout

```
peagent-analysis/
├── peagent/               model architecture + sequence/attribution/cell-group helpers
├── config/species.py      per-species paths + genome/metadata conventions
├── demo/                  simulated mini-atlas + run_demo.sh
├── 01_training/           preprocess ACRs, train the model
├── 02_evaluation/         per-peak AUROC, per-group PCC/AUROC/AUPR, latent NMI
├── 03_lexicon/            attribution maps, motif discovery, meta-pattern lexicon
├── 04_cooccurrence/       base-motif spatial co-occurrence
├── 05_synergy/            in-silico motif-pair synergy
├── 06_cross_species/      cross-species prediction and evaluation
├── 07_acr_evolution/      grass-conserved vs lineage-specific ACR grammar
└── 08_variant_effect/     REF/ALT variant scoring + caQTL benchmark
```

The analysis scripts import the **`peagent`** package in this repository (model
architecture, sequence/attribution/cell-group helpers, plotting defaults); run them
from the repository root and no separate install is needed.

## Analyses

| Step | Script(s) | What it does |
|------|-----------|--------------|
| **1. Training** | `01_training/01_preprocess_acrs.py`, `02_train_model.py` | Recentre ACRs on 1,344 bp windows, one-hot encode, split by ACR, and train the multitask CNN (Adam, binary cross-entropy, early stopping on validation AUROC). |
| **2. Evaluation** | `02_evaluation/01_per_peak_auroc.py` | AUROC / AUPR per held-out ACR across cell-type groups. |
| | `02_evaluation/02_per_group_metrics.py` | Per-cell-group profile PCC, AUROC and AUPR with a shuffled-label null. |
| | `02_evaluation/03_latent_nmi.py` | UMAP + Leiden on the learned cell embedding; NMI against annotated cell types. |
| **3. Lexicon** | `03_lexicon/01_attribution_maps.py` | Bulk and cell-type contribution maps from bottleneck ISM. |
| | `03_lexicon/02_modisco_finemo.py` | TF-MoDISco motif discovery + Fi-NeMo genome-wide instance calling. |
| | `03_lexicon/03_metapattern_compendium.py` | Cross-species meta-pattern clustering (CPM-Leiden). |
| | `03_lexicon/04_classify_lexicon.py` | Classify meta-patterns into base / composite / unresolved / noisy. |
| **4. Co-occurrence** | `04_cooccurrence/01_seqlet_cooccurrence.py` | Base-motif instance-pair co-occurrence and spacing summaries. |
| **5. Synergy** | `05_synergy/01_insilico_synergy.py` | In-silico marginalization test for motif-pair synergy. |
| **6. Cross-species** | `06_cross_species/01_cross_species_predict.py` | Score one species' ACRs with another species' model; semantic cell-type matching. |
| **7. ACR evolution** | `07_acr_evolution/01_acr_conservation.py` | Grass-conserved vs rice-specific ACR contribution + conservation/selection intersection. |
| **8. Variant effect** | `08_variant_effect/01_score_variants.py` | REF/ALT Δlogit scoring and fine-mapped-caQTL discrimination benchmark. |

## System requirements

- **Operating system:** Linux x86_64; tested on Rocky Linux 9.5.
- **Software:** Python 3.8 with the exact versions in [`environment.yml`](environment.yml)
  (tested: python 3.8.20, tensorflow 2.13.1, numpy 1.24.3, scipy 1.10.1, pandas 2.0.3, h5py 3.11.0,
  scikit-learn 1.3.2, statsmodels 0.14.1, matplotlib 3.7.5, pysam 0.23.3, anndata 0.9.2, scanpy 1.9.8,
  leidenalg 0.10.2, igraph 0.11.8, umap-learn 0.5.7); conda 24.11.3.
  Motif discovery (`03_lexicon/02-03`) used a separate Python 3.11 environment with modisco-lite 2.4.0,
  finemo-gpu 0.30 and MotifCompendium ([github.com/kundajelab/MotifCompendium](https://github.com/kundajelab/MotifCompendium)).
- **Hardware:** the demo needs only a standard computer (4 CPU cores, 16 GB RAM; no GPU).
  Training on the full atlases used one NVIDIA H100 80 GB GPU and 256 GB RAM per species
  (`01_training/run_train.sbatch`).

## Installation

```bash
git clone https://github.com/YAOJ-bioin/peagent-analysis
cd peagent-analysis
conda env create -f environment.yml
conda activate peagent
```

Typical install time: about 3-5 minutes (184 s measured with empty package caches).

## Demo

`demo/data/demo/` is a small simulated atlas (2 Mb genome, 1,200 ACRs, 240 cells in 4 tissue x
cell-type groups). Each group's ACRs carry a different planted motif. It is produced by
`demo/make_demo_data.py`. The demo preprocesses it, trains the PEAgent model for 20 epochs on CPU and
evaluates held-out ACRs (steps 1 and 2 above):

```bash
bash demo/run_demo.sh            # writes demo_output/
```

Expected output: `demo_output/model/best_model.h5`, and in `demo_output/eval/` the files
`demo_per_peak_auroc.csv` and `demo_per_group_metrics.csv` (reference copies in
[`demo/expected_output/`](demo/expected_output/)). The final lines print approximately

```
best val AUROC: 0.89
demo: n=49  median AUROC=1.000  median AUPR=1.000
demo: 4 groups  median PCC=0.94  median AUPR=0.97 (shuffle 0.20)
```

Training is not seeded, so the numbers vary slightly between runs (median PCC 0.94-0.97 in our tests).
Run time: about 5-6 minutes on 4 CPU cores (275-356 s measured).

## Instructions for use

1. Place each species' inputs under one directory and point `$PEAGENT_DATA_ROOT` at it:
   `<species>/atlas.h5ad` (cells x ACRs, binary, with tissue and cell-type columns in `obs`),
   `<species>/peaks.bed`, `<species>/genome.fa` (+ `.fai`) and, after training, `<species>/model.h5`.
   Column names, assemblies and chromosome patterns are set in [`config/species.py`](config/species.py);
   add an entry there for a new species.
2. Run the scripts in the order of the Analyses table; every script documents its arguments with `--help`.
   GPU jobs use the Slurm template `01_training/run_train.sbatch`.
3. **Reproduction:** the three atlases are public (see Data availability in the paper). The trained
   models are distributed with the PEAgent tool,
   [github.com/YAOJ-bioin/peagent-agent](https://github.com/YAOJ-bioin/peagent-agent).

## References

- Yuan H. & Kelley D. R. scBasset. *Nat. Methods* (2022).
- Kelley D. R. *et al.* Basenji. *Genome Res.* (2018); Kelley D. R. *et al.* Basset. *Genome Res.* (2016).
- Shrikumar A. *et al.* TF-MoDISco (2020); Fi-NeMo motif instance calling.
- Gupta S. *et al.* TomTom / MEME Suite. *Genome Biol.* (2007).
- Traag V. A. *et al.* Leiden clustering. *Sci. Rep.* (2019).

## Citation

This code accompanies:

> Yao J. *et al.* **Sequence-based modeling of plant epigenomes reveals
> cell-type-specific *cis*-regulatory grammar.** *Under review* (2026).

Software metadata is in `CITATION.cff`. Archived releases are on Zenodo:
[10.5281/zenodo.21495954](https://doi.org/10.5281/zenodo.21495954) (all versions).

## License

MIT — see `LICENSE`.
