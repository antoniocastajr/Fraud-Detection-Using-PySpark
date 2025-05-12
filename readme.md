# 🚨 Fraud Detection Using PySpark

This repository shows a implementation of a **fraud detection system using PySpark**. Fraud detection in financial transactions is the process of identifying suspicious or unauthorized activities —such as stolen credit card use, account takeovers, or synthetic identities— within banking or payment systems. It represents a critical area of financial security, as global losses due to payment fraud are projected to exceed $40 billion annually. 

The objective is to develop a scalable fraud detection system capable of identifying suspicious transactions in large datasets. I use **ensemble methods** to improve classification performance and address class imbalance challenges, which are common in fraud datasets.

---

## 📌 Key Features

✅ **Fraud detection pipeline built with PySpark**, optimized for large-scale financial transaction data

✅ **Exploratory Data Analysis (EDA)** with efficient PySpark DataFrame queries and visual summaries

✅ **Feature engineering and transformation**, including label encoding, type casting, and imbalance analysis

✅ **Model training with multiple classifiers**: XGBoost, LightGBM, and ensemble combinations with Logistic Regression

✅ **Ensemble learning** strategy to improve detection performance using model stacking

---

## 📂 Repository Structure

```
📁 fraud-detection-pyspark/
│
├── 📁 src/
|   └── 📘 fraud_detection.ipynb                # Jupyter Notebook implementing fraud detection pipeline in PySpark
|   └── 📘 fraudTest.csv                        # Dataset used for testing
|   └── 📘 fraudTrain.csv                       # Dataset used for training
│
├── 📁 Theory/
│   ├── 📁 SQL/
│   │   └── 📖 SQL.pdf                          # SQL foundations with personal annotations
│   │   └── 📖 SQL-Manual.pdf                   # SQL manual with personal annotations
│   └── 📁 Spark/
│   │   └── 📖 PySpark.pdf                      # Spark theory + Spark SQL & DataFrame API guide with custom notations
│   │   └── 📖 spark-the-definitive-guide.pdf   # SQL manual with personal annotations
│
└── 📄 README.md                                # This file
```

---

## 🧠 Theory & Notation Reference

### 📖 PySpark.pdf
- Spark execution model: Driver, Executor, Cluster Manager  
- Lazy evaluation, transformations, actions, DAG  
- DataFrame API vs SQL API  
- Machine learning with MLlib: `StringIndexer`, `VectorAssembler`, Pipelines  
- Personal annotations highlighting key concepts

### 📖 SQL.pdf
- SELECT, WHERE, GROUP BY, ORDER BY basics  
- Advanced filtering: LIKE, BETWEEN, IS NULL  
- Aggregation functions & column aliases  
- Join types and precedence rules  

These theory files contain annotated summaries, acting as a quick reference for Spark and SQL essentials during implementation.

---

## 🧪 Notebook Overview: `fraud_detection.ipynb`

This notebook contains a complete fraud detection workflow:

### 🔍 Data Loading & Exploration

**Objective:** Load and inspect the data to understand its structure and balance.

- Load the dataset into a **Spark DataFrame** using `spark.read.csv`.
- Inspect the **main columns and their data types** with `.printSchema()`.
- Display basic statistics using `.describe()` and `.select().show()` to check distributions.
- Evaluate **class imbalance** between fraudulent and non-fraudulent transactions.
- Check for **missing values** and possible imputation strategies.

---

### 🔢 Numerical Feature Engineering

**Objective:** Create meaningful numerical features to improve model performance.

- Generate new features, for example: **Age** of the client, **distance** between client's zip and merchant's zip or **average amount** by client, merchant, and category.
- Evaluate **correlation between each numerical feature and fraud labels**.
- **Drop low-correlation and non-informative numerical variables** to reduce noise and dimensionality.

---

### 🔣 Categorical Feature Engineering

**Objective:** Analyze and transform categorical features to capture fraud patterns.

- Visualize the **distribution of fraud cases** across categorical variables.
- Engineer new features such as:
  - Frequency encodings.
  - Category-based fraud ratios.
- Check their **statistical correlation with fraud** using cross-tabulations.
- Remove **irrelevant or noisy categorical variables**.

---

### 🏗 Pipeline Preparation

**Objective:** Design and implement flexible ML pipelines for various model types.

Four main pipelines are defined:
1. **XGBoost Pipeline:** Gradient boosting using `xgboost-spark`.
2. **LightGBM Pipeline:** Fast and scalable boosting via `lightgbm-spark`.
3. **Hybrid Pipeline:** XGBoost used for feature extraction, followed by Logistic Regression for classification.
4. **Hybrid Pipeline:** LightGBM used for feature extraction, followed by Logistic Regression for classification.

---

### ⚙️ Model Training & Prediction

**Objective:** Train and evaluate each model efficiently using distributed resources.

- Models are trained on a **Databricks Spark cluster**.
- Full pipeline training takes **~20–25 minutes in total** due to scale and feature complexity.
- Predictions are generated using `.transform()` on the test set.
- Model evaluation includes:
  - Confusion Matrix
  - Precision, Recall, F1-Score

---

## 📊 Results

The following models were trained and compared:

### ✅ **XGBoost** 
- Powerful gradient boosting model known for high recall.
- **Accuracy**: **98.70%**  
- **Precision**: **0.2227**  
- **Recall**: **94.73%**  
- **F1-score**: **0.3606**  
- 🟢 Very aggressive — captures almost all fraud but results in many false positives.  
- ⚠️ However, this model may be the best for fraud detection since **false negatives are very costly** in real-world financial systems.

---

### ✅ **LightGBM**
- Fast and memory-efficient gradient boosting framework.
- **Accuracy**: **99.74%**  
- **Precision**: **0.6400**  
- **Recall**: **71.84%**  
- **F1-score**: **0.6769**  
- 🟢 Offers a more balanced trade-off between fraud detection and minimizing false positives compared to XGBoost.

---

### ✅ **XGBoost + Logistic Regression**
- Combines XGBoost for feature extraction with Logistic Regression for refined classification.
- **Accuracy**: **99.84%**  
- **Precision**: **0.7909**  
- **Recall**: **78.46%**  
- **F1-score**: **0.7877**  
- 🟢 Improved both precision and recall, effectively reducing false alarms while maintaining strong fraud detection.

---

### ✅ **LightGBM + Logistic Regression**
- Uses LightGBM-derived features passed into a Logistic Regression model.
- **Accuracy**: **99.82%**  
- **Precision**: **0.9002**  
- **Recall**: **58.88%**  
- **F1-score**: **0.7120**  
- 🟢 Good overall performance with excellent specificity, though recall is lower than XGBoost-based models.

---

## 📈 Conclusion

XGBoost alone demonstrates very aggressive behavior in detecting fraud, identifying nearly all fraudulent cases (high recall), but at the cost of a high number of false positives (low precision). To address this, I combined tree-based models (XGBoost, LightGBM) with Logistic Regression. This hybrid ensemble strategy improves precision by refining predictions through a linear layer while preserving the ability to capture non-linear patterns. As a result, the stacked models offer better overall fraud detection, fewer false alarms, and greater robustness, all while maintaining a balance between interpretability and predictive power.

