
# Spam Detection Using Machine Learning

This repository contains the implementation and analysis of a machine learning-based spam detection system. The project evaluates different models and feature extraction techniques to classify messages as spam or ham (not spam). It provides insights into model performance, feature representation, and error analysis.

---

## **Project Overview**

The main objective of this project is to build and evaluate machine learning models to classify text messages into two categories: **spam** and **ham**. The project explores multiple techniques, including **TF-IDF**, **Bag of Words (BOW)**, and **Multinomial Naive Bayes (MNB)**, while focusing on key evaluation metrics like precision, recall, and F1-score.

### **Key Features**
- Preprocessing of raw text messages.
- Feature extraction using **TF-IDF** and **Bag of Words (BOW)**.
- Model comparison: **Logistic Regression (TF-IDF and BOW)** and **Multinomial Naive Bayes (MNB)**.
- Evaluation using confusion matrices, classification reports, and cross-validation.
- Error analysis for misclassified messages.

---

## **Data**

The dataset used for this project contains labeled text messages categorized as either **spam** or **ham**. The dataset includes a balanced mix of both categories to ensure robust model evaluation.

---

## **Models Evaluated**

1. **Logistic Regression with TF-IDF**:
   - Balanced performance with **94% precision** and **91% recall** for spam.
   - Suitable for general-purpose spam detection.

2. **Logistic Regression with Bag of Words (BOW)**:
   - High precision (**98%**) but lower recall (**90%**) for spam.
   - Ideal for applications where avoiding false positives is critical.

3. **Multinomial Naive Bayes (MNB)**:
   - High recall (**95%**) for spam but slightly lower precision (**91%**).
   - Best for scenarios prioritizing spam detection over false positives.

---

## **Performance Comparison**

| Model                     | Precision (Spam) | Recall (Spam) | Accuracy |
|---------------------------|------------------|---------------|-----------------|----------|
| Logistic Regression (TF-IDF) | 94%              | 91%           | 98%      |
| Logistic Regression (BOW)    | 98%              | 90%           | 98%      |
| Multinomial Naive Bayes       | 91%              | 95%           | 98%      |

**Key Insight**: The choice of model depends on whether precision (avoiding false positives) or recall (capturing more spam) is prioritized.

---

## **Error Analysis**

An error analysis of misclassified messages revealed:
- Some spam messages were misclassified as ham due to insufficient explicit keywords or ambiguity in phrasing.
- Certain ham messages were misclassified as spam due to excessive use of numeric data or terms resembling spam-like patterns.

This analysis highlights potential improvements in:
1. Text preprocessing (e.g., removing numeric sequences or certain stopwords).
2. Adjusting the model to better handle edge cases.

---

## **Cross-Validation Insights**

The project uses **cross-validation** during hyperparameter tuning for logistic regression models. The variation in accuracy across folds was visualized to identify:
- Stability of the models across different parameter combinations.
- Minimal signs of overfitting or underfitting, confirming robust generalization.

---

## **Usage**

### **Requirements**
- Python 3.8+
- Libraries: `scikit-learn`, `numpy`, `pandas`, `matplotlib`
