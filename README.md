# 📧 Spam Email Detection

A machine learning project that classifies emails as **Spam** or **Not Spam** using TF-IDF text features. Four classical ML models were trained and compared, with the best performer (SVM) deployed as an interactive web app.

**🚀 Live Demo:** [spam-email-detection-hmxaaaokv3yvxkmw7crgwo.streamlit.app](https://spam-email-detection-hmxaaaokv3yvxkmw7crgwo.streamlit.app/)

---

## Overview

Emails are preprocessed and vectorized using **TF-IDF**, then fed into four different classifiers. The models are evaluated on accuracy, precision, recall, and F1-score, and the best-performing model is saved for use in the Streamlit app.

## Project Structure

```
Spam-email-detection/
├── app.py                     # Streamlit web app
├── data/
│   └── emails.csv             # Training dataset
├── models/
│   ├── best_model.pkl         # Trained SVM model
│   └── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
├── results/                   # Confusion matrix plots for each model
├── src/
│   ├── preprocessing.py       # Data loading & TF-IDF vectorization
│   ├── models.py              # Model definitions
│   ├── train.py                # Training pipeline
│   └── evaluation.py          # Evaluation & confusion matrix utilities
└── requirements.txt
```

## Models

Four classifiers were trained and compared on the same TF-IDF features:

- **Support Vector Machine (LinearSVC)**
- **Random Forest**
- **Naive Bayes (Multinomial)**
- **Logistic Regression**

## Results

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1-score |
|----------------------|:--------:|:---------:|:------:|:--------:|
| **SVM**              | 0.9881   | 0.9868    | 0.9676 | 0.9771   |
| Logistic Regression   | 0.9744   | 0.9929    | 0.9094 | 0.9493   |
| Random Forest         | 0.9727   | 1.0000    | 0.8964 | 0.9454   |
| Naive Bayes           | 0.8925   | 1.0000    | 0.5922 | 0.7439   |

**SVM achieved the best overall performance** and was selected as the production model (`models/best_model.pkl`).

> **Note on dataset update:** The initial model was trained on the original dataset and achieved 99.21% test accuracy. During external testing, several false positives and false negatives were identified. The dataset was subsequently expanded with modern spam examples and legitimate security notifications to improve real-world generalization. The results below reflect this expanded dataset.

### Detailed Classification Reports

<details>
<summary><strong>SVM</strong></summary>

| Class        | Precision | Recall | F1-score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| 0 (Not Spam) | 0.99      | 1.00   | 0.99     | 863     |
| 1 (Spam)     | 0.99      | 0.97   | 0.98     | 309     |
| **Accuracy** |           |        | **0.99** | 1172    |
| Macro avg    | 0.99      | 0.98   | 0.98     | 1172    |
| Weighted avg | 0.99      | 0.99   | 0.99     | 1172    |

</details>

<details>
<summary><strong>Random Forest</strong></summary>

| Class        | Precision | Recall | F1-score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| 0 (Not Spam) | 0.96      | 1.00   | 0.98     | 863     |
| 1 (Spam)     | 1.00      | 0.90   | 0.95     | 309     |
| **Accuracy** |           |        | **0.97** | 1172    |
| Macro avg    | 0.98      | 0.95   | 0.96     | 1172    |
| Weighted avg | 0.97      | 0.97   | 0.97     | 1172    |

</details>

<details>
<summary><strong>Naive Bayes</strong></summary>

| Class        | Precision | Recall | F1-score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| 0 (Not Spam) | 0.87      | 1.00   | 0.93     | 863     |
| 1 (Spam)     | 1.00      | 0.59   | 0.74     | 309     |
| **Accuracy** |           |        | **0.89** | 1172    |
| Macro avg    | 0.94      | 0.80   | 0.84     | 1172    |
| Weighted avg | 0.91      | 0.89   | 0.88     | 1172    |

</details>

<details>
<summary><strong>Logistic Regression</strong></summary>

| Class        | Precision | Recall | F1-score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| 0 (Not Spam) | 0.97      | 1.00   | 0.98     | 863     |
| 1 (Spam)     | 0.99      | 0.91   | 0.95     | 309     |
| **Accuracy** |           |        | **0.97** | 1172    |
| Macro avg    | 0.98      | 0.95   | 0.97     | 1172    |
| Weighted avg | 0.97      | 0.97   | 0.97     | 1172    |

</details>

Confusion matrices for all four models are available in the [`results/`](./results) folder.

## ⚠️ Limitation

The model relies solely on textual features extracted using TF-IDF. It does not analyze sender information, URLs, metadata, or email headers, which can lead to false positives for legitimate security notifications.

## Getting Started

### Installation

```bash
git clone https://github.com/Sourith2008/Spam-email-detection.git
cd Spam-email-detection
pip install -r requirements.txt
```

### Train the models

```bash
cd src
python train.py
```

This trains all four models, prints classification reports, saves confusion matrix plots to `results/`, and exports the best model (SVM) and TF-IDF vectorizer to `models/`.

### Run the app locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, paste in an email, and click **Detect spam**.

## Tech Stack

- Python, scikit-learn, pandas, NumPy
- TF-IDF (`TfidfVectorizer`) for feature extraction
- Streamlit for the web interface
- Matplotlib / Seaborn for visualizations
