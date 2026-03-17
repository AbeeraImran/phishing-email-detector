# Phishing Email Detector

A comparative study of four machine learning and deep learning architectures for detecting phishing emails. Built as part of an NLP course, this project walks through the full pipeline from raw text to rigorous evaluation — covering Logistic Regression, a Feedforward Neural Network, a Vanilla RNN, and an LSTM.

---

## What This Project Does

Takes raw email text and classifies it as **phishing** or **safe**. Four models of increasing complexity are trained on the same dataset, split, and evaluation pipeline so their performance can be directly compared. The goal is not just to build models but to understand *why* each one performs the way it does.

---

## Models Implemented

| Model | Type | Input Representation |
|---|---|---|
| Logistic Regression | Classical ML | TF-IDF vectors |
| Feedforward Neural Network | Deep Learning | TF-IDF vectors |
| Vanilla RNN | Sequential DL | Token embeddings |
| LSTM | Sequential DL | Token embeddings |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9732 | 0.9522 | 0.9808 | 0.9663 | 0.9961 | 2.05s |
| FNN | 0.9726 | 0.9427 | **0.9904** | 0.9660 | 0.9958 | 9.86s |
| Vanilla RNN | 0.7403 | 0.6561 | 0.7100 | 0.6820 | 0.7529 | 7.78s |
| LSTM | 0.9506 | 0.9187 | 0.9590 | 0.9384 | 0.9890 | 29.31s |

**Recommended for deployment: FNN** — highest Recall (0.9904), meaning only 7 out of 731 phishing emails in the test set were missed. In a security-critical system, minimising False Negatives (missed phishing emails) is the primary objective.

---

## Dataset

**Phishing Emails — Subha Journal**
- Source: [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- Size: 18,634 emails after cleaning
- Classes: Safe (60.8%) / Phishing (39.2%)
- Split: 80% train / 10% validation / 10% test (stratified)

---

## How to Run

**1. Open in Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**2. Enable GPU**
```
Runtime → Change runtime type → T4 GPU
```

**3. Download the dataset**

The notebook uses `kagglehub` to download automatically:
```python
import kagglehub
path = kagglehub.dataset_download("subhajournal/phishingemails")
```

**4. Run all cells**
```
Runtime → Run all
```

No manual file uploads required.

---

## Key Findings

**Logistic Regression is a surprisingly strong baseline.** With TF-IDF + bigrams it achieves 0.9663 F1 and trains in 2 seconds — faster than every neural model by a large margin. For this dataset, the TF-IDF feature space appears nearly saturated.

**The FNN edges out LR on Recall.** It catches 7 more phishing emails than LR (724 vs 717 out of 731) at the cost of 5x longer training time. Whether this tradeoff is worth it depends on the security requirements of the deployment environment.

**The Vanilla RNN fails badly.** F1 of 0.6820 and AUC of 0.7529 — barely better than random for some metrics. The vanishing gradient problem prevents it from learning dependencies across sequences longer than ~20 tokens, and with max_seq_len=200 this is a fundamental limitation. Gradient norm analysis confirmed near-zero gradients in the embedding layer.

**The LSTM recovers dramatically.** By replacing the RNN cell with LSTM gates, False Negatives drop from 212 to 30 — 182 fewer missed phishing emails. This directly demonstrates the value of the forget/input/output gate mechanism for long-range sequence modelling.

**Adversarial test revealed a real vulnerability.** A manually crafted phishing email written in formal corporate language was classified as safe with 90% confidence by the best model (FNN). This confirms the model has learned surface vocabulary patterns rather than semantic intent, and motivates transformer-based approaches for future work.

---

## Evaluation Highlights

- Confusion matrices with real-world cost interpretation for all 4 models
- ROC curves for all models on one axes
- Precision-Recall curves with Average Precision scores
- Brier score + reliability diagrams (calibration check)
- Threshold sensitivity analysis (0.3 → 0.7)
- Error analysis on 10 misclassified examples with confidence scores
- Gradient norm tracking to visualise vanishing gradients in the RNN

---

## Tech Stack

```
Python 3.10
PyTorch
TensorFlow / Keras (tokenizer only)
scikit-learn
pandas / numpy
matplotlib / seaborn
kagglehub
Google Colab (T4 GPU)
```

---

## Limitations and Future Work

- Text-only models cannot access URL structure, sender metadata, or email headers — all strong phishing signals
- max_seq_len=200 truncates emails at the ~65th percentile word count — longer sequences could improve RNN and LSTM performance
- Fine-tuning BERT or RoBERTa would likely outperform all four models significantly
- Ensemble of LR + LSTM predictions could combine the strengths of both approaches
- Lowering the classification threshold to 0.3–0.4 increases Recall further at a manageable Precision cost

---

## Author

Built for academic purposes as part of an NLP fundamentals course.
Feel free to fork, explore, or build on it.
