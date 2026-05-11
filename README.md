# 🤖 Midterm Machine Learning — End-to-End ML Pipeline

> **Implementasi End-to-End Machine Learning Pipeline** mencakup Klasifikasi, Regresi, dan Clustering dengan Optuna Hyperparameter Tuning & MLflow Experiment Tracking

---

## 📋 Identitas

| | |
|---|---|
| **Nama** | Mohammad Fauzi Hadiwijaya |
| **NIM** | 101032300044 |
| **Kelas** | TK-47-04 |

---

## 📌 Deskripsi Repository

Repository ini berisi **3 project end-to-end machine learning** yang dikerjakan sebagai tugas UTS mata kuliah Machine Learning. Setiap project mencakup pipeline lengkap mulai dari preprocessing, training, hyperparameter tuning (Optuna), experiment tracking (MLflow), hingga evaluasi model.

---

## 📁 Struktur Repository

```
midterm-machine-learning/
│
├── fraud_detection.ipynb          # Task 1: Fraud Detection (Klasifikasi)
├── regression_song_year.ipynb     # Task 2: Song Year Prediction (Regresi)
├── clustering_customer.ipynb      # Task 3: Customer Clustering
├── submission.csv                 # Output prediksi fraud detection
└── README.md                      # Dokumentasi ini
```

---

## 📂 Task Overview

### Task 1 — 🔍 Fraud Detection (Klasifikasi)
**Dataset**: IEEE-CIS Fraud Detection (`train_transaction.csv`, `test_transaction.csv`)

Membangun model klasifikasi untuk memprediksi probabilitas transaksi online bersifat fraudulent (`isFraud`).

**Pipeline:**
- EDA & analisis class imbalance (~3.5% fraud)
- Preprocessing: drop kolom missing >80%, label encoding, imputasi median
- Feature engineering: log transform, jam & hari transaksi
- Handle imbalance: SMOTE
- Training 5 model: LR baseline → RF baseline → XGB baseline → RF Optuna → XGB Optuna
- Evaluasi: AUC-ROC, F1-Score, Accuracy, Precision, Recall
- MLflow tracking semua run

**Model yang digunakan:**
| Model | Keterangan |
|---|---|
| Logistic Regression | Baseline linear |
| Random Forest (baseline) | Default params |
| XGBoost (baseline) | Default params |
| Random Forest (Optuna) | 20 trials tuning |
| XGBoost (Optuna) | 20 trials tuning |

**Hasil Evaluasi (Validation Set):**
| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| LR Baseline | 0.7433 | 0.0935 | 0.6579 |
| RF Baseline | 0.8754 | 0.3848 | 0.9547 |
| XGB Baseline | 0.9272 | 0.3417 | 0.9248 |
| RF Optuna | 0.9084 | 0.5693 | 0.9830 |
| XGB Optuna | 0.9366 | 0.7119 | 0.9872 |

> 🏆 Best: XGB_optuna (AUC-ROC: 0.9366)

---

### Task 2 — 🎵 Song Year Prediction (Regresi)
**Dataset**: `midterm-regresi-dataset.csv`

Membangun model regresi untuk memprediksi tahun rilis lagu berdasarkan fitur audio (timbre dan karakteristik sinyal musik).

**Pipeline:**
- EDA: distribusi tahun rilis, korelasi fitur dengan target
- Preprocessing: outlier clipping (IQR x3), imputasi median, standard scaling
- Training 5 model: Ridge → RF baseline → XGB baseline → RF Optuna → XGB Optuna
- Evaluasi: RMSE, MAE, R², MSE
- MLflow tracking semua run
- **LIME** untuk interpretasi prediksi model secara lokal

**Model yang digunakan:**
| Model | Keterangan |
|---|---|
| Ridge Regression | Baseline linear |
| Random Forest (baseline) | Default params |
| XGBoost (baseline) | Default params |
| Random Forest (Optuna) | 20 trials tuning |
| XGBoost (Optuna) | 20 trials tuning |

**Hasil Evaluasi (Validation Set):**
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Baseline | — | — | — |
| RF Baseline | — | — | — |
| XGB Baseline | — | — | — |
| RF Optuna | — | — | — |
| XGB Optuna | — | — | — |

> *Isi tabel setelah notebook dijalankan*

---

### Task 3 — 👥 Customer Clustering
**Dataset**: `clusteringmidterm.csv`

Membangun pipeline clustering untuk mengelompokkan customer berdasarkan perilaku penggunaan kartu kredit.

**Pipeline:**
- EDA: distribusi fitur, heatmap korelasi, deteksi outlier
- Preprocessing: imputasi median, outlier clipping, standard scaling, PCA 2D
- Menentukan k optimal: Elbow Method + Silhouette Score
- Training 3 model: K-Means, Hierarchical Clustering (+ Dendrogram), DBSCAN
- Evaluasi: Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score
- Visualisasi: PCA 2D, Heatmap profil cluster, Radar chart

**Hasil Evaluasi:**
| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| K-Means | — | — | — |
| Hierarchical | — | — | — |
| DBSCAN | — | — | — |

> *Isi tabel setelah notebook dijalankan*

**Interpretasi Cluster (K-Means, k=2):**
| Cluster | Nama Segmen | Karakteristik |
|---|---|---|
| 0 | Premium Active Customer | Balance tinggi, purchases sangat tinggi (2499), credit limit besar (6717), rajin membayar |
| 1 | Cash-Dependent Customer | Cash advance tinggi (889), purchases rendah (328), sering tarik tunai, risiko lebih tinggi |

---

## ⚙️ Tools & Framework

| Tool | Kegunaan |
|---|---|
| `scikit-learn` | Preprocessing, models, evaluasi |
| `xgboost` | XGBoost classifier & regressor |
| `imbalanced-learn` | SMOTE untuk class imbalance |
| `optuna` | Hyperparameter tuning otomatis (TPE) |
| `mlflow` | Experiment tracking & model logging |
| `lime` | Interpretasi prediksi model (Task 2) |
| `yellowbrick` | Visualisasi clustering |

---

## 🚀 Cara Menjalankan

1. **Clone repo ini**
   ```bash
   git clone https://github.com/fauziihadiwijaya/midterm-machine-learning.git
   ```

2. **Upload dataset ke Google Drive** folder `midterm-m1`:
   - `train_transaction.csv` & `test_transaction.csv` (Task 1)
   - `midterm-regresi-dataset.csv` (Task 2)
   - `clusteringmidterm.csv` (Task 3)

3. **Buka notebook di Google Colab**
   - Upload notebook yang diinginkan
   - Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
   - Sesuaikan path dataset di cell Load Dataset
   - `Runtime → Run all`

---

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn optuna mlflow lime yellowbrick
```
