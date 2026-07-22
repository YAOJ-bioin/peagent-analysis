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
├── config/species.py      per-species paths + genome/metadata conventions
├── 01_training/           preprocess ACRs, train the model
├── 02_evaluation/         per-peak AUROC, per-group PCC/AUROC/AUPR, latent NMI
├── 03_lexicon/            attribution maps, motif discovery, meta-pattern lexicon
├── 04_cooccurrence/       base-motif spatial co-occurrence
├── 05_synergy/            in-silico motif-pair synergy
├── 06_cross_species/      cross-species prediction and evaluation
├── 07_acr_evolution/      grass-conserved vs lineage-specific ACR grammar
└── 08_variant_effect/     REF/ALT variant scoring + caQTL benchmark
```

The analysis scripts import the **`peagent`** package (model architecture,
sequence/attribution/cell-group helpers, plotting defaults), which is released
as a separate package and installed alongside this repository (see *Installation*).

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

Software metadata is in `CITATION.cff`; a Zenodo DOI is minted for the archived
release (see `docs/RELEASE.md`).

## License

MIT — see `LICENSE`.
