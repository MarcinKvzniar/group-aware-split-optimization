# Metaheuristic Optimization of Group-Aware Stratified Splitting

> **Course:** Optimization Methods - Final Project

---

## Problem Description

Many real-world datasets are naturally **hierarchical**: multiple images may belong to one patient, or many patches to a single whole-slide image (WSI). Splitting such data into Train / Validation / Test sets creates two conflicting constraints:

- **Group-level isolation** -- all items from the same group must land in the same split (no data leakage).
- **Class balance** -- each split should mirror the global class distribution, which is hard when groups vary in size and rare classes may be concentrated in just a few groups.

Standard scikit-learn splitters handle one constraint or the other, never both simultaneously. This project formulates the problem as a variant of the **multidimensional Multiple Knapsack Problem** and evaluates metaheuristic optimizers against deterministic baselines.

### Cost function

Weighted MAPE summed over all *(split x class)* pairs:

$$
\mathcal{L} = \sum_{s,\,c} w_c \cdot \frac{|\text{actual}_{s,c} - \text{target}_{s,c}|}{\text{target}_{s,c} + \varepsilon}
$$

where $w_c \propto 1/f_c$ gives higher penalty to rare classes.

## Datasets

*No download needed - for optimization all datasets are summarized in groups.pkl rerpresentation*

### BCSS - Breast Cancer Semantic Segmentation
151 whole-slide images tiled into 512 x 512 patches (8 768 tiles total), 21 tissue classes.

- Download from [Google Drive](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss) and extract to `./datasets/bcss/`
- Citation: Amgad et al., *Bioinformatics* 35(18), 2019. DOI: [10.1093/bioinformatics/btz083](https://doi.org/10.1093/bioinformatics/btz083)

### CelebA - Large-Scale Face Attributes
202 599 celebrity face images grouped by identity (10 177 unique persons), 40 binary attributes.

- Download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and extract to `./datasets/celeb-faces/`
- Also requires `identity_CelebA.txt` (available in the same Kaggle package)
- Citation: Liu et al., *ICCV*, December 2015.

### ISIC 2020 - Melanoma Classification
33 126 dermoscopy images from 2 056 patients, 9 diagnosis classes (1.76 % melanoma).

- Download **only `train.csv`** from the [Kaggle competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data?select=train.csv) (requires competition sign-up) and place it in `./datasets/isic2020/`
- License: CC BY-NC 4.0 - academic use only
- Citation: Zawacki et al., *SIIM-ISIC Melanoma Classification*, Kaggle 2020.
