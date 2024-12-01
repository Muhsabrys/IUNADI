# IUNADI at NADI 2023 shared task: Country-level Arabic Dialect Classification in Tweets for the Shared Task NADI 2023

## Overview
This repository contains the code and resources for our paper: ["IUNADI at NADI 2023 shared task: Country-level Arabic Dialect Classification in Tweets for the Shared Task NADI 2023"](https://aclanthology.org/2023.arabicnlp-1.72.pdf). Our work describes an approach to classify Arabic dialects from tweets as part of the NADI 2023 shared task. The dataset includes tweets from 18 Arab countries, and the goal is to classify tweets at the country level.

## Abstract
We explored various machine learning models, including:
- Traditional models such as Multinomial Naive Bayes, SVM, XGBoost, AdaBoost, and Random Forests.
- Transformer-based models such as AraBERTv2-Base, AraBERTv2-Large, and CAMeLBERT-Mix DID MADAR.

The best performance was achieved using AraBERTv2-Large, which outperformed traditional classifiers and other transformer-based models.

## Methodology

### Dataset
The dataset comprises:
- 23.4K tweets from 18 countries.
- Division: 18K for training, 1.8K for development, and 3.6K for testing.
- Additional datasets: NADI 2020, NADI 2021, and MADAR.

### Pre-processing Steps
1. Remove diacritics.
2. Normalize Hamza and Lam-Alif.
3. Remove Kashida (elongation of Arabic letters).
4. Remove punctuation.
5. Correct common spelling errors.
6. Map numerical labels to corresponding country names.

### Models and Training
- **Traditional Models**: Multinomial Naive Bayes, SVM, XGBoost, AdaBoost, and Random Forests.
- **Transformer Models**: AraBERTv2-Base, AraBERTv2-Large, CAMeLBERT-Mix DID MADAR.

**Training Details:**
- Fine-tuning was performed using HuggingFace's `transformers` library.
- Key configurations:
  - `adam_epsilon`: 1e-8
  - `learning_rate`: 2e-5
  - Batch size: 16 (up to 64 for high-memory GPUs).
  - Maximum sequence length: 128.
  - Training epochs: 3.
  - 5-fold cross-validation for robustness.

### Evaluation Metrics
- **Macro-averaged F1-score**: The official metric.
- Additional metrics: Precision, Recall, and Accuracy.

## Results
### Official Task Results
We submitted AraBERTv2-Large as our final model. The model achieved:
- **F1-score**: 70.22
- **Accuracy**: 70.78

### Development Set Results
| Model                        | F1   | Accuracy | Precision | Recall |
|------------------------------|-------|----------|-----------|--------|
| AraBERTv2-Large              | 0.71  | 0.71     | 0.71      | 0.71   |
| AraBERTv2-Base               | 0.71  | 0.71     | 0.70      | 0.71   |
| CAMeLBERT-Mix DID MADAR      | 0.71  | 0.71     | 0.70      | 0.71   |
| XGBoost                      | 0.52  | 0.51     | 0.60      | 0.51   |
| Random Forest                | 0.43  | 0.42     | 0.51      | 0.42   |
| Multinomial Naive Bayes      | 0.41  | 0.45     | 0.73      | 0.45   |
| SVM                          | 0.39  | 0.40     | 0.58      | 0.40   |
| AdaBoost                     | 0.18  | 0.18     | 0.50      | 0.18   |

## Challenges
- **Dialect Variation**: Arabic dialects exhibit significant variation and code-switching with Standard Arabic.
- **Limited Data**: Expanding the dataset could further improve model performance.

## Future Work
- Investigate ensemble approaches combining transformer-based models.
- Explore advanced fine-tuning techniques such as domain adaptation and multi-task learning.
- Evaluate the impact of additional pre-processing techniques, including tokenization, stop word removal, and stemming.

## Citation
If you use our work, please cite:
```
@inproceedings{hatekar-abdo-2023-iunadi,
    title = "{IUNADI} at {NADI} 2023 shared task: Country-level {A}rabic Dialect Classification in Tweets for the Shared Task {NADI} 2023",
    author = "Hatekar, Yash  and
      Abdo, Muhammad",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.72",
    doi = "10.18653/v1/2023.arabicnlp-1.72",
    pages = "665--669",
    abstract = "In this paper, we describe our participation in the NADI2023 shared task for the classification of Arabic dialects in tweets. For training, evaluation, and testing purposes, a primary dataset comprising tweets from 18 Arab countries is provided, along with three older datasets. The main objective is to develop a model capable of classifying tweets from these 18 countries. We outline our approach, which leverages various machine learning models. Our experiments demonstrate that large language models, particularly Arabertv2-Large, Arabertv2-Base, and CAMeLBERT-Mix DID MADAR, consistently outperform traditional methods such as SVM, XGBOOST, Multinomial Naive Bayes, AdaBoost, and Random Forests.",
}
```

## Maintainer
- [Muhammed S. Abdo](https://www.linkedin.com/in/muhsabrys/)
