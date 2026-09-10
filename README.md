# Computing-Research-Project
Machine Learning and Natural Language Processing Based Fake News Detection System for Social Media Platforms
# Fake News Detection System

A reproducible machine learning pipeline for detecting fake and real news articles using traditional machine learning, deep learning, and transformer-based NLP models.

## 📌 Overview

This project implements an end-to-end **Fake News Detection System** using news article titles and text. The pipeline includes data validation, duplicate and leakage auditing, text preprocessing, stratified cross-validation, traditional machine learning models, an LSTM neural network, a fine-tuned DistilBERT model, and an OOF-weighted ensemble.

The project is designed with an emphasis on **reproducibility, leakage prevention, and reliable model evaluation**.

## 🎯 Objectives

* Detect whether a news article is **fake or real**.
* Compare traditional machine learning approaches with deep learning and transformer models.
* Apply leakage-safe preprocessing and model validation.
* Evaluate models using multiple classification metrics.
* Build an ensemble using out-of-fold (OOF) predictions.
* Provide reproducible results using fixed random seeds and saved checkpoints.

## 📊 Dataset

The system uses a `train.tsv` dataset containing news articles.

The raw dataset contains:

* **30,000 records**
* **6 columns**
* `title`
* `text`
* `subject`
* `date`
* `label`
* `Unnamed: 0`

The target variable is binary:

* `0` — Fake News
* `1` — Real News

Initial class distribution:

| Label |  Count |
| ----- | -----: |
| 0     | 15,478 |
| 1     | 14,522 |

After removing exact duplicate texts and short/empty articles, the final modelling population contains **27,309 articles**.

## 🔍 Data Quality & Leakage Audit

Before modelling, the pipeline performs several data-quality checks:

* Missing-value detection
* Exact duplicate detection
* Duplicate title detection
* Conflicting-label detection
* Near-duplicate text analysis
* Class-distribution analysis
* Date-range analysis

The dataset initially contained **2,683 exact duplicate texts**, which were removed.

No identical texts with conflicting labels were found.

A near-duplicate audit was also performed using character-level TF-IDF and cosine similarity.

## 🧹 Text Preprocessing

The preprocessing pipeline performs:

1. Conversion to lowercase
2. URL removal
3. Email-address removal
4. Removal of non-alphabetic characters
5. Whitespace normalisation
6. Tokenisation
7. English stop-word removal
8. Word lemmatisation
9. Combination of cleaned title and article text

Articles with fewer than 50 characters after preprocessing are removed.

The final input to the models is a combination of:

```text
cleaned title + cleaned article text
```

## 🧪 Train/Test Strategy

A fixed **80:20 stratified split** is used.

| Split       | Samples |
| ----------- | ------: |
| Development |  21,847 |
| Test        |   5,462 |

The random seed is fixed at:

```text
42
```

The test set remains locked until final evaluation.

## 🤖 Models

The project evaluates several different approaches.

### Traditional Machine Learning

#### Logistic Regression

Uses TF-IDF features with:

* Maximum features: 5,000
* N-gram range: 1–2
* Minimum document frequency: 2
* Maximum document frequency: 0.95

#### Random Forest

Configuration includes:

* 200 estimators
* Maximum depth: 25
* Minimum samples split: 5
* Minimum samples leaf: 2

#### Support Vector Machine

A linear SVM is trained using TF-IDF features with:

```text
kernel = linear
C = 1.0
```

### Deep Learning

#### Bidirectional LSTM

The LSTM architecture contains:

* Embedding layer
* Spatial dropout
* Bidirectional LSTM with 64 units
* Bidirectional LSTM with 32 units
* Dense layer with 64 units
* Dropout
* Sigmoid output layer

Configuration:

```text
Vocabulary size: up to 10,000
Sequence length: 250
Embedding dimension: 100
Batch size: 32
```

Early stopping is used during training.

### Transformer Model

#### Fine-tuned DistilBERT

The project fine-tunes:

```text
distilbert-base-uncased
```

Configuration:

```text
Maximum sequence length: 256
Epochs: 2
Batch size: 16
```

The model is trained on the development set and evaluated once on the locked test set.

## 🔗 OOF-Weighted Ensemble

An ensemble is constructed using out-of-fold predictions from:

* Logistic Regression
* Random Forest
* SVM
* LSTM

The ensemble weights are optimised using constrained optimisation so that:

* All weights are non-negative
* Weights sum to 1
* Optimisation is based on log loss

The final learned weights were approximately:

| Model               | Weight |
| ------------------- | -----: |
| Logistic Regression | 0.0000 |
| Random Forest       | 0.0002 |
| SVM                 | 0.8314 |
| LSTM                | 0.1685 |

This indicates that the SVM contributed the largest share to the final ensemble.

## 📈 Cross-Validation Results

Traditional machine learning and LSTM models were evaluated using **5-fold stratified cross-validation**.

| Model               |        Accuracy |              F1 |         ROC-AUC |
| ------------------- | --------------: | --------------: | --------------: |
| SVM                 | 0.9902 ± 0.0015 | 0.9907 ± 0.0014 | 0.9991 ± 0.0004 |
| Logistic Regression | 0.9838 ± 0.0013 | 0.9848 ± 0.0012 | 0.9984 ± 0.0005 |
| LSTM                | 0.9808 ± 0.0042 | 0.9818 ± 0.0042 | 0.9983 ± 0.0007 |
| Random Forest       | 0.9755 ± 0.0016 | 0.9771 ± 0.0015 | 0.9980 ± 0.0004 |

### Ensemble Cross-Validation

The honest outer-fold evaluation of the OOF-weighted ensemble produced:

| Metric    |   Mean |     SD |
| --------- | -----: | -----: |
| Accuracy  | 0.9918 | 0.0012 |
| Precision | 0.9914 | 0.0019 |
| Recall    | 0.9930 | 0.0012 |
| F1        | 0.9922 | 0.0012 |
| ROC-AUC   | 0.9995 | 0.0002 |
| Log Loss  | 0.0270 | 0.0025 |

## 🏆 Final Test Results

The final evaluation was performed on the previously untouched 20% test set.

| Model                 |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |   Log Loss |
| --------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| **DistilBERT**        | **0.9934** | **0.9910** | **0.9965** | **0.9938** |     0.9988 |     0.0357 |
| OOF-Weighted Ensemble |     0.9901 |     0.9883 |     0.9931 |     0.9907 | **0.9992** | **0.0288** |
| SVM                   |     0.9894 |     0.9872 |     0.9927 |     0.9900 |     0.9987 |     0.0383 |
| LSTM                  |     0.9848 |     0.9865 |     0.9847 |     0.9856 |     0.9985 |     0.0456 |
| Logistic Regression   |     0.9828 |     0.9755 |     0.9924 |     0.9838 |     0.9981 |     0.0851 |
| Random Forest         |     0.9749 |     0.9611 |     0.9927 |     0.9766 |     0.9981 |     0.1953 |

### Key Result

**Fine-tuned DistilBERT achieved the highest final test accuracy and F1 score:**

* Accuracy: **99.34%**
* Precision: **99.10%**
* Recall: **99.65%**
* F1 Score: **99.38%**

The OOF-weighted ensemble achieved the lowest log loss and the highest ROC-AUC among the evaluated final models.

## 📉 Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Log Loss
* Confusion Matrix
* ROC Curves
* Classification Report

Additional error analysis and visualisations are generated during the pipeline.

## 🔄 Reproducibility

Reproducibility is a major component of the project.

The pipeline uses:

```text
Random Seed = 42
```

Deterministic settings are configured for:

* Python
* NumPy
* TensorFlow
* PyTorch
* CUDA operations where supported

The environment used for the recorded run includes:

```text
Python        3.13.15
NumPy         2.1.3
Pandas        2.2.3
Scikit-learn 1.6.1
TensorFlow    2.20.0
PyTorch       2.11.0+cu128
Transformers  5.15.1
SciPy         1.16.3
NLTK          3.9.1
```

## 💾 Checkpointing & Outputs

The pipeline automatically creates project directories for:

```text
checkpoints/
models/
results/
figures/
cv/
logs/
config/
```

Examples of generated outputs include:

```text
models/
├── logistic_regression_final.joblib
├── random_forest_final.joblib
├── svm_final.joblib
└── distilbert_final/

cv/
├── traditional_cv_fold_results.csv
├── traditional_cv_summary.csv
├── lstm_cv_fold_results.csv
├── ensemble_outer_fold_results.csv
└── ensemble_outer_cv_summary.csv

results/
├── final_test_results.csv
├── ensemble_weights.csv
├── ensemble_weights.json
├── bert_test_proba.npy
├── cv_report_table.csv
└── ensemble_cv_report.csv
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* PyTorch
* Hugging Face Transformers
* NLTK
* SciPy
* Matplotlib
* Seaborn
* Joblib
* tqdm
* Google Colab / Google Drive

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/rajagopalarao/Computing-Research-Project.git)
cd YOUR_REPOSITORY
```

### 2. Add the dataset

Place the required dataset at:

```text
train.tsv
```

The notebook also searches several Google Drive and Colab locations for the dataset.

### 3. Install dependencies

The notebook automatically checks for missing Python packages and installs them when necessary.

Alternatively, install the main dependencies manually:

```bash
pip install numpy pandas scikit-learn tensorflow transformers torch nltk matplotlib seaborn joblib scipy tqdm
```

### 4. Run the notebook

Open:

```text
C5050526_Final_corrected(2).ipynb
```

in Google Colab or a compatible Jupyter environment.

Run the cells sequentially from top to bottom.

## 📁 Suggested Repository Structure

```text
fake-news-detection/
│
├── C5050526_Final_corrected(2).ipynb
├── train.tsv
├── README.md
│
├── models/
├── results/
├── figures/
├── cv/
├── checkpoints/
├── logs/
└── config/
```

## ⚠️ Important Notes

The reported performance is based on the specific dataset and experimental setup used in this project.

The notebook explicitly separates:

1. Cross-validation performance
2. Honest outer-fold ensemble performance
3. Final evaluation on the untouched test set

The higher final test performance should therefore not be interpreted as proof of universal generalisation.

The dataset also contains news from a historical period, so performance on contemporary news or news from different sources may differ.

## 🔬 Methodology Summary

```text
Raw News Dataset
       │
       ▼
Data Validation
       │
       ▼
Duplicate & Leakage Audit
       │
       ▼
Text Cleaning & Preprocessing
       │
       ▼
80/20 Stratified Split
       │
       ├───────────────┐
       ▼               ▼
 Traditional ML       Deep Learning
       │               │
 ┌─────┼─────┐        LSTM
 │     │     │
LR   RF    SVM
 └─────┼─────┘
       │
       ▼
OOF Predictions
       │
       ▼
Weighted Ensemble
       │
       ├───────────────┐
       ▼               ▼
 Ensemble Test      DistilBERT
 Evaluation         Fine-tuning
       │               │
       └───────┬───────┘
               ▼
        Final Test Results
```

## 📌 Conclusion

This project demonstrates a complete NLP-based fake news classification workflow, comparing conventional TF-IDF machine learning models with an LSTM network and a fine-tuned DistilBERT transformer.

The experiments show that **DistilBERT achieved the strongest final test classification performance**, while the **OOF-weighted ensemble provided very strong ROC-AUC and log-loss performance**.

The project also demonstrates the importance of duplicate removal, leakage auditing, stratified validation, fixed random seeds, out-of-fold predictions, and separate locked test evaluation when building reproducible machine learning experiments.

## 👤 Author

**Rajagopala Rao Bandaru**

GitHub: `https://github.com/rajagopalarao`

---

⭐ If you found this project useful, consider giving the repository a star!
