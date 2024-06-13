# MEXASER

This repository is intended to provide a collection of the files related to the paper titled: 

"MEXASER: Metaheuristics and Explainable AI for Speech Emotion Recognition".

### About the Project

The paper presents MEXASER, a novel framework designed to optimize feature selection for Speech Emotion Recognition (SER). It uses metaheuristic algorithms, such as Genetic Algorithm, Artificial Ecosystem Based Optimization, and Water Cycle Algorithm, to enhance classification accuracy. The framework’s efficacy is validated across four prominent speech emotion recognition datasets (Ravdess, Savee, Emodb, Tess), and selected features are analyzed using Shapley values to ascertain their significance in achieving high accuracy.

### Built With

Our code relies on the following resources/libraries:

- mealpy
- shap
- opensmile
- sklearn
- & other standard ML python libraries

### Setup

Python version used is Python 3.9.12

Run this command to install all required libraries

```bash
pip install -r requirements.txt

```
## data

1. `features_652.json`: Extracted features on the RAVDESS dataset utilized in previous works.
   
The remaining datasets are all publicly available and can be acquired as follows :

*Note : to match our task, features must be extracted using the opensmile library with the ComParE_2016 feature set*

1. RAVDESS : https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio 
2. TESS : https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
3. SAVEE : https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
4. EMODB : https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb

## notebooks
1. `beeswarm_heatmaps.ipynb` : Jupyter notebook analysing features selected based on shapley values.
2. `pipeline_nb.ipynb` : Notebook version of `example_pipeline.py` with feature extraction code.
3. `feat_exp.ipynb` : Explainability of selected features through frequency analysis plots.

## scripts

1. `example_pipeline.py`: Example implementation of our accuracy pipeline on the 'TESS' dataset using the BASE GA optimizer.
2. `sadness_f1.py`: Experiment on optimizing single class weighted f1 score.
3. `with_prev_features` : Demonstration of how the new pipeline outperforms previous work, even utilizing the same feature set.

## Compute Resources Utilized

For training and testing our models, we make use of the following compute resources:

- **Specs:**
  - Processor: 32 × 2 cores AMD EPYC 50375
  - RAM: 1 TB
  - GPUs: 8x NVIDIA A100 SXM4 80GB
