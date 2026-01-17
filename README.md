# Heart Disease Prediction & Machine Learning Benchmark

This project explores various Machine Learning and Deep Learning techniques to predict heart disease based on clinical patient data. It includes a comprehensive pipeline from data preprocessing and feature engineering to hyperparameter optimization and model evaluation.

## 🚀 Project Overview
The goal of this project is to develop a robust classifier that identifies whether a patient has heart disease based on medical attributes. The project compares classical ML models against a deep learning approach to determine the most effective predictive method.

## 📊 Dataset Features
The dataset includes several clinical indicators used as input features:
* **Demographics:** Age, Gender.
* **Vitals:** Blood Pressure, Heart Rate.
* **Laboratory Results:** Cholesterol levels.
* **Engineered Features:**
    * `AgeBucket`: Categorical grouping of age ranges.
    * `Chol_per_Age`: Ratio of cholesterol to age.
    * `HR_pct_max`: Heart rate relative to the theoretical maximum ($220 - Age$).

## 🛠️ Technical Workflow

### 1. Data Analysis & Visualization
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and correlations using Seaborn heatmaps.
* **Dimensionality Reduction:** Utilized **Principal Component Analysis (PCA)** to visualize class separation in 2D and analyze explained variance.



### 2. Model Implementation
I implemented and compared several industry-standard algorithms:
* **Linear Models:** Logistic Regression (with L1/L2 regularization).
* **Tree-Based Models:** Decision Tree, Random Forest, and Gradient Boosting Classifiers.
* **Distance/Probabilistic:** K-Nearest Neighbors (KNN) and Naive Bayes.
* **Deep Learning:** A multi-layer **Sequential Neural Network** built with TensorFlow/Keras using ReLU and Sigmoid activation functions.

### 3. Optimization & Evaluation
* **Preprocessing:** Data was scaled using `StandardScaler` and categorical variables were handled via `OneHotEncoder`.
* **Hyperparameter Tuning:** Automated search using `GridSearchCV` and `RandomizedSearchCV` to find optimal model parameters.
* **Validation:** Conducted **5-fold Cross-Validation** to ensure the models generalize well to unseen data.
* **Metrics:** Evaluation focused on Accuracy, F1-Score, and **ROC-AUC curves**.



## 📈 Key Results
The project provides a comparative analysis of all models, highlighting:
* Model accuracy comparisons via bar plots.
* Confusion matrices to analyze False Positives and False Negatives.
* ROC-AUC analysis to measure the trade-off between sensitivity and specificity.

## 💻 Requirements
To run this project, you will need:
* Python 3.10+
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-Learn
* TensorFlow

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
