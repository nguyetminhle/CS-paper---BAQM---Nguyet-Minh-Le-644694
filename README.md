# MSMRP: Duplicate Product Detection using LSH and MSM

## Project Overview
This project implements **MSMRP**, a scalable duplicate product detection pipeline for e-commerce data. The method builds on the LSH-MSM framework and extends it with enhanced data cleaning, model-word extraction, and domain-specific exclusion rules to improve precision while reducing computational cost.

The pipeline is designed to detect duplicate television products across multiple web shops by:
- Cleaning and normalizing product attributes (brand, resolution, screen size)
- Extracting model words and constructing binary product representations
- Generating candidate pairs using MinHashing and Locality Sensitive Hashing (LSH)
- Verifying candidates using the Multi-component Similarity Method (MSM)
- Evaluating performance using bootstrapped experiments

---

## Project Structure

The codebase is organized into modular components, each responsible for a specific stage of the pipeline:

- `main.py`: runs the full pipeline with bootstrap evaluation
- `data.py`: data preprocessing
- `binary_signature_vector.py`: model-word extraction and binary vector construction
- `minhash_cal.py`: MinHash signature computation
- `lsh_cal.py`: LSH candidate generation
- `sim_cal.py`: similarity computation formulas
- `msm_cal.py`: MSM similarity algorithm and exclusion rules
- `evaluation_cal.py`: evaluation metrics and performance calculation
---

## How the Pipeline Works

### Data Cleaning
Product descriptions are cleaned and normalized. Brand names, resolution indicators, and screen sizes are standardized to ensure consistent comparison.

### Binary Vector Construction
Model words are extracted from product titles and attributes. Each product is represented as a binary vector indicating the presence of these model words.

### Candidate Generation (Minhasing + LSH)
Binary vectors are compressed using MinHashing and partitioned into bands using LSH to generate a reduced set of candidate duplicate pairs.

### Verification (MSM)
Candidate pairs are scored using MSM, which combines title similarity, feature-value similarity, and model-word overlap. Additional exclusion rules remove implausible matches early.

### Evaluation
Performance is measured using **F1**, **F1\***, **Pair Completeness (PC)**, and **Pair Quality (PQ)** under a bootstrapped evaluation framework.

---

## How to Run the Code

### Requirements
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `itertools` 
- `collections` 
- `math`
- `random`

### Running the Pipeline
From the project directory, run:
```bash
python main.py
