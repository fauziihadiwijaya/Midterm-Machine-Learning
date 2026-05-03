# 🔍 Midterm Machine Learning — Fraud Detection

> **End-to-End Machine Learning Pipeline untuk Deteksi Transaksi Fraudulent**
> dengan Optuna Hyperparameter Tuning & MLflow Experiment Tracking

---

## 📋 Identitas

| | |
|---|---|
| **Nama** | Mohammad Fauzi Hadiwijaya |
| **NIM** | 101032300044 |
| **Kelas** | TK-47-04 |

---

## 📌 Deskripsi Proyek

Repository ini berisi implementasi **end-to-end machine learning pipeline** untuk mendeteksi transaksi online yang bersifat fraudulent. Dataset yang digunakan adalah **IEEE-CIS Fraud Detection** dari Kaggle.

### 🎯 Tujuan
Membangun model klasifikasi yang memprediksi **probabilitas** sebuah transaksi bersifat fraud (`isFraud`), dievaluasi menggunakan metrik **AUC-ROC**.

---

## 📁 Struktur Repository

```
midterm-machine-learning/
│
├── fraud_detection_ml_v2.ipynb   # Notebook utama: pipeline lengkap + Optuna + MLflow
├── submission.csv                # Output prediksi probabilitas fraud untuk test set
└── README.md                     # Dokumentasi ini
```

---

## 🔄 Pipeline Overview

```
Raw Data (train_transaction.csv + test_transaction.csv)
   │
   ▼
Exploratory Data Analysis (EDA)
   │  - Distribusi target (class imbalance ~3.5% fraud)
   │  - Missing value analysis
   │  - Distribusi TransactionAmt per kelas
   ▼
Data Preprocessing & Feature Engineering
   │  - Drop kolom missing > 80%
   │  - Feature baru: log(TransactionAmt), jam, hari, desimal amount
   │  - Label Encoding untuk fitur kategorik
   │  - Imputasi median untuk missing values
   ▼
Handle Class Imbalance
   │  - SMOTE (sampling_strategy=0.2)
   │  - class_weight / scale_pos_weight pada model
   ▼
MLflow Setup → Experiment: "fraud-detection-ml"
   │
   ▼
Baseline Model Training (+ MLflow logging otomatis)
   │  - Logistic Regression
   │  - Random Forest
   │  - XGBoost
   ▼
Hyperparameter Tuning dengan Optuna (20 trials per model)
   │  - Random Forest Optuna → log ke MLflow
   │  - XGBoost Optuna      → log ke MLflow
   ▼
Evaluasi & Perbandingan (5 model total)
   │  - AUC-ROC, F1, Accuracy, Precision, Recall
   │  - ROC Curves overlay
   │  - Confusion Matrices
   │  - Feature Importance
   ▼
MLflow Tracking Summary
   ▼
Generate submission.csv
```

---

## 🤖 Model yang Digunakan

| Model | Tipe | Keterangan |
|---|---|---|
| **Logistic Regression** | Baseline | Model linear; cepat, mudah diinterpretasi |
| **Random Forest (baseline)** | Ensemble | 200 pohon; default hyperparameter |
| **XGBoost (baseline)** | Gradient Boosting | Default hyperparameter |
| **Random Forest (Optuna)** | Ensemble | Hyperparameter dioptimasi Optuna (20 trials) |
| **XGBoost (Optuna)** | Gradient Boosting | Hyperparameter dioptimasi Optuna (20 trials) |

---

## ⚙️ Tools & Framework

| Tool | Kegunaan |
|---|---|
| `scikit-learn` | Preprocessing, baseline models, evaluasi |
| `xgboost` | XGBoost classifier |
| `imbalanced-learn` | SMOTE untuk class imbalance |
| **`optuna`** | Hyperparameter tuning otomatis (TPE algorithm) |
| **`mlflow`** | Experiment tracking: log params, metrics, model artifact |

---

## 📊 Hasil Evaluasi (Validation Set)

> *Tabel ini diisi setelah menjalankan notebook*

| Model | AUC-ROC | F1-Score | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| LR Baseline |  0.7433 | 0.0935 | 0.6579 | 0.0502 | 0.6895 |
| RF Baseline | — | — | — | — | — |
| XGB Baseline | — | — | — | — | — |
| **RF Optuna** | **—** | **—** | **—** | **—** | **—** |
| **XGB Optuna** | **—** | **—** | **—** | **—** | **—** |

**Metrik utama: AUC-ROC** — dipilih karena dataset sangat imbalanced (~3.5% fraud).

---

## 🚀 Cara Menjalankan

1. **Clone repo ini**
   ```bash
   git clone https://github.com/[username]/midterm-machine-learning.git
   ```

2. **Download dataset** dari [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   - File: `train_transaction.csv`, `test_transaction.csv`
   - Upload ke Google Drive: `MyDrive/midterm-ml/`

3. **Buka di Google Colab**
   - Upload `fraud_detection_ml_v2.ipynb`
   - Sesuaikan `TRAIN_PATH` & `TEST_PATH` di cell Load Dataset
   - Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
   - Run All

4. **Lihat MLflow UI** (opsional, jika run lokal)
   ```bash
   mlflow ui
   # Buka browser: http://127.0.0.1:5000
   ```

---

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost optuna mlflow
```
