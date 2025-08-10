# Materials Discovery


## Project Overview

The project presents an end-to-end machine learning and active learning framework for materials discovery. The system integrates robust data preprocessing, feature engineering, predictive modelling, uncertainity quantification, and Bayesian Optimization along with active learning to accelerate the identification of promising materials with desired properties.


## Table of Contents

• [Project Overview](#project_overview)

• [Dataset Description](#Dataset_Description)

• [Code Overview](#Code_Overview)

• [Technologies Used](#Technologies_used)

• [Installation Guide](#Installation_guide)

• [Folder Structure](#Folder_Structure)

• [LICENSE](#LICENSE)


## Dataset Description

This datatset contains structured data designed to train, evaluate, and validate machine learning models for predictive analytics. Key features of the dataset are :-

• ``Name                     ``:- name of matrials

• ``Formula                  ``:- Molecular formula of materials.

• ``Spacegroup               ``:- space group of crystal structure, which defines its symmetry                                      properties based on International Tables for Crystallography.

• ``nelements                ``:- number of distinct elements in compund.

• ``nsites                   ``:- total  number of atomic sites in crystal's unit cell (inclding                                    repeated atoms due to symmetry).

• ``energy_per_atom          ``:- total calculated energy of material normalized per atom 
                                  (eV/atom).This value is derived from density functional theory                                   (DFTcalculations and reflects the stability of structure                                         thermodynamically.

•``formation_energy_per_atom``:- formation energy of material normalized per atom (eV/atom).It
                                  measures energy change when compund forms from its constituent
                                  elements; lower value generally indicates higher stability.

• ``band_gap                ``:- energy difference between valence band maximum & conduction band
                                (eV). Critical property influences electrical conductivity, small
                                band gaps corresponds to semiconductors,while zero band indicates
                                metallic behavior.

• ``volume_per_atom         ``:- volume occupied by each atom in the material(Å³/atom).It reflect
                                  atomic packing and can affect density and other structural
                                  properties.

• ``magnetization_per_atom  ``:- net magnetic moment of material normalized per atom (μB/atom,                                     Bohr magnetons per atom). It determines magnetic ordering of 
                                   material (e.g.,ferromagnetic, antiferromagnetic).

• ``atomic_volume_per_atom  ``:- calculated atomic volume per atom (Å³/atom) based on atomic                                       structure model. This may differ ** volume_per_atom depending
                                  on structural distortions or defects.

• ``volume_deviation        ``:- percentage deviation between calculated ** volume_per_atom and                                    expected atomic_volume_per_atom .It indicates strain, lattice                                    distortion, or non-ideal packing.

### Dataset    :- https://figshare.com/articles/dataset/CFID_OQMD_460k/13055333?file=24981170


## Workflow Overview

In this section, there is complete machine learning workflow for material property prediction, from preprocessing, through model training, hyperparameter optimization, uncertainity quantification, and active learning with Bayesian optimization.


### Preprocessing

In this section, handles the missing values,performs feature engineering, outlier removal, categorical encoding, feature scaling, and correlation visualization.

![Correlation Heatmap](https://github.com/Sakura-hack01/Materials-Discovery/blob/66384592f8eed44dd635881d2dfa187e6e3ca2a6/Preprocessed%20Dataset/output.png)
• This image shows that dataset is cleaned, normalilized, and transformed for optimal model performance.

### ML modelling

Here,the use of XGBoost with Optuna for hyperparameter tuning to minimize MSE and evaluate with R² score.Includes SHAP explainability to initerpret feature importance.

![Summary Plot](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/ML%20modelling/output.png)
• Hyperparameters are optimized to improve predictive accuracy, while SHAP identifies most impactful features.

### Bayesian Optimization

It leverages BoTorch and GPyTorch to suggest top candidate based on Expected Improvement (EI).

![Acquition function plot](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Bayesian%20Optimization/output.png)
• This identifies most promising unexplored candidates, aiding target experimentation.

### Uncertainity- Aware Deep Learning Model

It implements a heteroscedastic neural network ensemble to predict both mean, variance of outputs, enabling uncertainity quantification.

### Active Learning

It combines Bayesian optimization with the sampling of uncertainity to iteratively select and label most informative data points.

![Active learning iteration visualization](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Active%20Learning/output1.png)
![Active learning iteration visualization](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Active%20Learning/output2.png)
![Active learning iteration visualization](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Active%20Learning/output3.png)
![Active learning iteration visualization](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Active%20Learning/output4.png)
![Active learning iteration visualization](https://github.com/Sakura-hack01/Materials-Discovery/blob/22a901ebb80f99e04369938882e84c25c7bf57b5/Active%20Learning/output5.png)
• It reduces labeling costs and accelerates learning by focusing on uncertain and high value samples.


## Technologies Used

• Python : version 3.9-3.11

• GPU : optional but recommended for deep learning components

• Libraries and Frameworks : numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, os, torch, ijson, xgboost, optuna, shap, botorch, gpytorch, flask


## Installation Guide

Install all dependies via pip:
`` bash
pip install numpy ijson xgboost optuna scikit-learn scipy seaborn shap matplotlib joblib torch botorch gpytorch flask ``
• For GPU acceleartion with PyTorch, visit official [Pytorch website](https://pytorch.org/get-started/locally/) to install the CUDA-enabled version.

### Clone the repository

``bash
https://github.com/Sakura-hack01/Materials-Discovery.git
cd Materials-Discovery``

### Running the application
We use flask here for web-based application, this is used when we use flask in CLI :-
``bash
flask run``
If in bash we go with :-
``bash
python app.py``


## Folder Structure

```
Materials-Discovery
|
|___Dataset
|   |
|   |__dataset.csv
|___Preprocessed Dataset
|   |
|   |__link_of_preprocessed_dataset.csv
|   |
|   |__preprocess_pipeline.py
|   |
|   |__output.png
|
|___ML modelling
|   |
|   |__modelling.py
|   |__output.png
|
|___Uncertainity-aware Deep Learning Model
|   |
|   |__uncertainity.py
|   
|___Bayesian Optimization
|   |
|   |__bayesian.py
|   |
|   |__output.png
|
|___Active Learning
|   |
|   |__active_learning.py
|   |
|   |__output1.png
|   |
|   |__output2.png
|   | 
|   |__output3.png
|   | 
|   |__output4.png
|   | 
|   |__output5.png
|   | 
|   |__output6.png
|   | 
|   |__output7.png
|   | 
|   |__output8.png
|   | 
|   |__output9.png
|   | 
|   |__output10.png
|
|___app.py
```


## License

This project is lincensed under MIT License - see the [LICENSE](LICENSE) file for details.
