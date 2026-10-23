# Polygenic gene-set tester for the Yao et al. MERFISH mouse brain

A Streamlit app for the MERFISH whole mouse brain atlas of Yao et al. (2023; Allen Brain Cell Atlas, donor
`C57BL6J-638850`, 500-gene panel). Enter a gene set; every cell gets an exact set-versus-complement AUROC from its
gene expression ranks, shown on 16 coronal sections in two rows of eight, anterior to posterior. The app also tests
whether the gene set is specifically expressed in each **cell type** and each **brain region**, using pseudobulk
profiles as in the [SEA-AD polygenic tester](https://github.com/leonfrench/SEA_AD_multi_polygenic_app). It is a
cell-based relative of the [spatial fetal app](https://github.com/leonfrench/spatial_fetal_app).

## Run locally

```bash
~/miniconda3/bin/conda create -y -p ./.conda-env -c conda-forge --override-channels python=3.12 pip
./.conda-env/bin/python -m pip install -r requirements-prep.txt pytest
./.conda-env/bin/streamlit run app.py
```

Open http://localhost:8501. `requirements.txt` is enough to run the app on the prepared data in
`data/merfish-638850/`; `requirements-prep.txt` adds polars and h5py for preprocessing.
`MERFISH_APP_DATA=/path/to/export` selects a different prepared dataset.

## Deploy on Streamlit Community Cloud

- Main file: `app.py`. Python 3.12 is the development version; the app and its tests also pass on Python 3.14 with a
  clean install of `requirements.txt` (the preprocessing tests are skipped there). `requirements.txt` is the dependency
  file Community Cloud installs; do not add an `environment.yml`, which it would install instead.
- The prepared data (337 MB; every file is under GitHub's 100 MB limit, the largest is 82 MB) is committed in plain git,
  so no Git LFS is needed.
- Memory: Community Cloud allows up to 2.7 GB. A local server held about 0.55 GB at rest and peaked at 1.2 GB while a
  browser session scored every cell and switched through all levels and colour modes; a second concurrent session did
  not raise the peak. Label columns load as categories, and the plot geometry is built once and shared by all sessions.
- The first page load sends about 25 MB of plot data to the browser.

## Using the app

- **Sections**: 16 of the 53 CCF-registered coronal sections, anterior to posterior (olfactory bulb to cerebellum and
  medulla), about 0.9 mm apart. Where the section nearest an even step is torn or partial (for example `.39`, `.35`,
  `.31`, `.27`, `.19`), an intact neighbour within 0.2 mm is used. Panels are labelled with the section number of
  `C57BL6J-638850.NN`. To change sections, reprocess with `--sections`.
- **Genes**: mouse symbols by default (matched case-insensitively). Human or rhesus macaque symbols are converted to
  mouse orthologs with the bundled HomoloGene snapshot (`data/homologene_to_human.csv.gz`); one-to-many orthologs are all
  kept and unmapped symbols are reported. An optional background is the complete universe, including the targets;
  ranks are recomputed inside it. **Gene matching** lists the targets used and any symbols left out, and its download
  gives every gene of the scoring universe with its role (target or background). The default gene set is the medial
  habenula markers on the panel (`Tac2 Chat Slc5a7 Chrnb3 Pou4f1`).
- **Colour cells by**: gene-set AUROC, cell type, region, cell type AUROC or region AUROC (the last two colour each cell
  by its label's specificity AUROC from the tables). The cell type level (class / subclass / supertype / cluster) and
  region level (division / structure / substructure) set both the colours and the tables.
- **Plot**: one panel per section, all sharing the same CCF-aligned (reconstructed) frame and scale in micrometres, on a
  black background, dorsal up. Dots are sized to the median nearest-cell spacing (~11 µm). AUROC ranges differ between
  sections, so gene-set AUROC colours are relative within each section: each is scaled between its own 1st and 99th
  percentiles, which are shown above its panel, and the legend is in arbitrary units. Hover gives the exact AUROC. The
  label AUROC modes (one pooled value per label) share one scale clipped at the 1st and 99th percentiles of the cells
  shown. The clip is `COLOUR_CLIP` in `app.py`. Colours only; the statistics are unchanged. Drag to pan, scroll to zoom,
  double-click to reset, hover for a cell's annotations and AUROC, click a legend entry to highlight that label, and save
  the current view as PNG.
- **Tables**: cell type and region expression specificity across all sections (AUROC, p-value and the
  Benjamini–Hochberg adjusted pFDR), with CSV downloads, plus a download of
  every cell's score once scoring completes. The tables use pseudobulk profiles, not cell scores, so they appear as soon
  as the gene set is valid. Tick a row's checkbox to colour the plot by that table's specificity AUROC (cell type or
  region) and highlight the label; untick it to return to the previous colour mode. A table with a parent level shows
  it as the second column.

## Data

Inputs live in `data/MERFISH/` (not version controlled), in the layout of the
[Allen Brain Cell Atlas bucket](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html). The expression
matrix, cell metadata and panel come with the dataset; the annotation tables (~258 MB) are downloaded from the bucket
when missing.

| File | Contents |
| --- | --- |
| `MERFISH-C57BL6J-638850/20230630/C57BL6J-638850-raw.h5ad` | 4,330,907 segmented cells × 550 barcodes, raw counts (CSR), rows grouped by section (59 sections) |
| `MERFISH-C57BL6J-638850/20230830/cell_metadata.csv` | 3,938,808 cells that passed QC and were mapped: `cell_label`, `brain_section_label`, `cluster_alias`, section `x, y, z` (mm) |
| `MERFISH-C57BL6J-638850/20230830/gene.csv` | The 550 barcodes in matrix order, with Ensembl IDs |
| `MERFISH-C57BL6J-638850-CCF/20231215/reconstructed_coordinates.csv` | 3,739,961 CCF-registered cells: aligned `x, y, z` (mm) and `parcellation_index` |
| `WMB-taxonomy/20231215/views/cluster_to_cluster_annotation_membership_{pivoted,color}.csv` | `cluster_alias` → class, subclass, supertype and cluster names and colours |
| `Allen-CCF-2020/20230630/views/parcellation_to_parcellation_term_membership_acronym.csv`, `parcellation_term.csv` | `parcellation_index` → division, structure and substructure acronyms; their names and colours |

What the inspection showed:

- Section `.01` (z = 0) is the most posterior (cerebellum and medulla) and `.69` (z = 15 mm) the most anterior
  (olfactory bulb); sections are 0.2 mm apart, with gaps. Sections `.01`–`.04`, `.68` and `.69` have no CCF registration.
- A cell is kept if it is in the cell metadata (QC-passed and mapped) and CCF-registered. 2,471 metadata cells (0.06%)
  have no row in the expression matrix and are dropped; the manifest reports them per section. For every kept cell the
  reconstructed z equals the metadata z.
- The panel has 500 genes and 50 `Blank-*` control barcodes. Raw counts are integers stored as floats. Kept cells have
  a median of 541 counts and 135 detected genes; 71.7% of cell × gene entries are zero.
- Both hemispheres carry the same CCF parcellation, so region pseudobulks use all cells. Cells outside the
  parcellation (1.6%) are `unassigned`.

## Preprocessing

```bash
./.conda-env/bin/python scripts/prepare_merfish_cells.py --force
```

`--raw` (default `data/MERFISH`) holds the inputs above and `--reference` (default `data/reference`) the MGI marker
lists, which are downloaded on the first run. The 16 sections take about 4 minutes, with peak memory of 4.4 GB; each
section's rows of the CSR matrix are read straight from the h5ad. `--force` replaces an existing output directory once
the new one is complete.

Steps:

1. **Cells**: cells in the metadata with a CCF registration, in expression-matrix order within each section; sections
   ordered anterior to posterior. Display coordinates are the reconstructed `x, y` in micrometres.
2. **Genes**: the universe is the 500 panel genes; the Blank barcodes are dropped, and their counts are kept per cell
   as `n_blank_counts`. Symbols are updated to current MGI symbols, then unambiguous withdrawn symbols (`Car4` → `Ca4`,
   `Fyb` → `Fyb1`, `Prrxl1` → `Drgx`); no synonym guessing. `Lhfp` is an ambiguous withdrawn symbol and keeps its panel
   symbol. Because every panel gene was chosen as a marker, the 14 lncRNAs and the `Ighm` gene segment stay in the
   universe.
3. **Ranks**: average ranks within each cell, with all undetected genes tied at the bottom, stored as doubled `uint16`
   values (format `sparse-doubled-average-ranks-v1`). Thirty-two random cells per section are checked against
   `scipy.stats.rankdata` and against a pairwise AUROC, and every batch is checked after a write/read round trip.
4. **Batches**: each section's cells are shuffled (seed 20261022) and split into equal batches of at most 16,384 cells
   (`--batch-cells`) and 12 million stored cell × gene entries (`--batch-entries`), in `progressive/<SECTION>/NNNNN.npz`.
5. **Annotations**: `annotations.json` holds labels and colours for each level. Cell types use the atlas taxonomy
   colours. Regions use Allen CCF colours, with near-identical colours (such as cortical layers) spread in lightness so
   they stay distinguishable; `unassigned` is grey rather than the atlas black, the plot background.

Output (`data/merfish-638850/`, 337 MB, of which 15 MB is pseudobulk):

| File | Contents |
| --- | --- |
| `manifest.json` | Universe size, per-section cell counts and ranges, batch paths, input checksums, mapping counts, validation |
| `cells.parquet` | `cell_index`, `slice` (section number), `cell_label`, `h5ad_row`, `brain_section_label`, `cluster_alias`, `average_correlation_score`, section `x, y` (µm), `z_mm`, reconstructed `rx, ry` (µm), `parcellation_index`, `neurotransmitter`, `cell_class`, `cell_subclass`, `cell_supertype`, `cell_cluster`, `region_division`, `region_structure`, `region_substructure`, `n_counts`, `n_genes_detected`, `n_blank_counts`, `draw_order`, `batch`, `batch_row` |
| `genes.parquet` | `gene_index`, symbol, MGI ID, name, MGI feature type, Ensembl ID, panel symbol, detection counts |
| `gene_detection.parquet` | Detected-cell counts per gene and section (for the sparsity note) |
| `pseudobulk/<SECTION>.npz` | For each annotation level: summed panel-gene counts per (label, gene) for the labels present (`<level>_counts`, `uint32`), their global label indices (`<level>_index`, into `manifest.json` `pseudobulk.labels`) and cell counts (`<level>_cells`) |
| `gene_mapping.parquet` | All 550 barcodes with their MGI mapping and status (`retained` or `control_barcode`) |
| `annotations.json` | Titles, labels, colours and region names for all seven annotation levels |

| Section | z (mm) | Segmented cells | Kept cells | Batches |
| --- | ---: | ---: | ---: | ---: |
| 67 | 14.2 | 31,293 | 29,243 | 2 |
| 64 | 13.0 | 23,833 | 22,269 | 2 |
| 62 | 12.2 | 35,538 | 33,129 | 3 |
| 59 | 11.4 | 40,133 | 37,480 | 3 |
| 56 | 10.8 | 60,473 | 57,160 | 4 |
| 52 | 9.8 | 98,696 | 93,186 | 6 |
| 47 | 8.8 | 87,885 | 80,676 | 5 |
| 44 | 8.0 | 128,812 | 112,033 | 7 |
| 40 | 7.2 | 114,251 | 104,461 | 7 |
| 36 | 6.4 | 123,292 | 115,675 | 8 |
| 32 | 5.6 | 104,317 | 93,442 | 6 |
| 28 | 4.6 | 87,724 | 79,114 | 5 |
| 24 | 3.8 | 74,831 | 61,570 | 4 |
| 15 | 2.8 | 87,076 | 80,305 | 5 |
| 09 | 1.6 | 75,870 | 69,210 | 5 |
| 05 | 0.8 | 50,990 | 45,896 | 3 |
| **Total** | | **1,225,014** | **1,114,849** | **75** |

The cell types are the whole mouse brain taxonomy: 34 classes, 338 subclasses, 1,201 supertypes and 5,322 clusters, of
which 34, 338, 1,193 and 5,041 occur in these sections. The regions are 25 divisions, 318 structures and 617
substructures.

## Methods

**Cell scores (plot).** For k target genes in a universe of G genes, a cell's score is
`AUROC = (sum of target ranks − k(k+1)/2) / (k(G − k))`, the probability that a target gene outranks a
non-target gene in that cell, with half credit for ties.

**Expression specificity (tables).** For the chosen level, each label's pseudobulk sums the transcript counts of its
cells across all sections and is converted to log₂(CPM + 1). Each gene is z-scored across the labels, removing its
average expression (genes that do not vary get z = 0). Within each label, genes are ranked by z-score (average ranks for
ties), and the AUROC compares the target genes with the rest of the universe: above 0.5 means the gene set is more
specific to that label than other genes. With a custom background, genes are ranked within that universe. Labels with
fewer than 20 cells are not profiled. Two-sided Mann–Whitney p-values over genes (normal approximation with tie and
continuity correction, matching `scipy.stats.mannwhitneyu`) are adjusted with Benjamini–Hochberg within each table
(pFDR).
Genes are not independent (co-expression), so the p-values are exploratory. This follows the
[SEA-AD polygenic tester](https://github.com/leonfrench/SEA_AD_multi_polygenic_app), which ranks genes by pseudobulk
z-scores in the same way; its p-values omit the tie correction.

## Tests

```bash
./.conda-env/bin/python -m pytest tests -q
node tests/test_cells.cjs
```

These cover rank encoding and scoring against SciPy and pairwise definitions, panel mapping and count reading,
pseudobulk sums, specificity z-scores and AUROCs against `mannwhitneyu`, ortholog mapping, the grid layouts, the
score transport protocol, the canvas helpers, and end-to-end app runs on `data/merfish-638850`.

## Data credit

Yao et al. *A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain.*
Nature 624, 317–332 (2023). [doi:10.1038/s41586-023-06812-z](https://doi.org/10.1038/s41586-023-06812-z).
Data, taxonomy and CCF registration: [Allen Brain Cell Atlas](https://brain-map.org/bkp/explore/abc-atlas).
Gene symbols: [MGI](https://www.informatics.jax.org/). Orthologs: HomoloGene.
