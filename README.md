# **Big Data Analysis Techniques and Tools – Practical Work Series** 

## **Project Overview**

This repository contains a series of eight practical assignments completed as part of the course *Technologies and Tools for Big Data Analysis*. Each practical work focuses on different aspects of data analysis, from basic Python programming and data visualization to advanced machine learning, clustering, regression, classification, ensemble methods, and association rule mining.

The assignments are implemented in Python using popular libraries such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `plotly`, `seaborn`, `statsmodels`, and others.


## **List of Practical Works**

### **Practical Work 1: Introduction to Python and Basic Data Operations**
- Installation of Python and setup of the environment.
- Writing programs to calculate the area of geometric shapes (triangle, rectangle, circle).
- Implementation of a calculator supporting basic and advanced operations.
- Reading numbers until their sum equals zero, then computing the sum of squares.
- Generating sequences based on input length.
- Working with lists and dictionaries to map values.
- Loading the California housing dataset from `sklearn`, performing basic data exploration, and filtering data.
- Text encoding using Morse code.
- User registration system simulation.
- File system access control simulation.


### **Practical Work 2: Data Visualization with Plotly, Matplotlib, and Dimensionality Reduction**
- Loading and describing multidimensional data (Life Expectancy dataset).
- Data preprocessing and cleaning.
- Building bar charts and pie charts using `plotly.graph_objs`.
- Creating line plots with `matplotlib` to analyze trends over time.
- Dimensionality reduction and visualization using **t-SNE** and **UMAP** on the MNIST dataset.
- Comparison of t-SNE and UMAP performance and visual clarity.


### **Practical Work 3: Statistical Analysis and Hypothesis Testing**
- Loading and exploring the `insurance.csv` dataset.
- Descriptive statistics and histogram plotting.
- Measures of central tendency and dispersion for BMI and charges.
- Box plot construction and interpretation.
- Verification of the Central Limit Theorem using sampling.
- Construction of 95% and 99% confidence intervals.
- Normality testing using KS-test and Q-Q plots.
- Loading and preprocessing COVID-19 data (`ECDCCases.csv`).
- Handling missing values and outliers.
- Duplicate detection and removal.
- Student’s t-test for comparing BMI across regions with Shapiro-Wilk and Bartlett tests.


### **Practical Work 4: Correlation, Linear Regression, and ANOVA**
- Calculating Pearson correlation between two variables and scatter plot visualization.
- Loading and preprocessing the Life Expectancy dataset.
- Building a correlation matrix and identifying the most correlated variable.
- Implementing linear regression manually and using `sklearn`.
- Gradient descent implementation for regression.
- One-way and two-way ANOVA tests to examine the effect of region and sex on BMI.
- Post-hoc Tukey tests and visualization.


### **Practical Work 5: Classification Algorithms**
- Loading and preprocessing heart disease dataset.
- Checking class balance with visualizations.
- Splitting data into training and test sets.
- Implementing and evaluating:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Model comparison using accuracy, precision, recall, and F1-score.


### **Practical Work 6: Clustering Algorithms**
- Loading and preprocessing customer personality analysis dataset.
- Feature engineering and normalization.
- Clustering using:
  - K-Means (with elbow method and silhouette score)
  - Agglomerative Clustering
  - DBSCAN
- Visualization of clusters in 3D space.
- Interpretation of customer segments based on income, spending, and family size.


### **Practical Work 7: Ensemble Learning**
- Using the same heart disease dataset for classification.
- Implementing:
  - **Bagging**: Multiple decision trees combined via mode voting.
  - **Boosting**: Random Forest with GridSearchCV and CatBoost.
- Comparing performance (R² and F1-score) and training time between bagging and boosting.


### **Practical Work 8: Association Rule Mining**
- Loading market basket optimization dataset (`Market_Basket_Optimisation.csv`).
- Visualizing item frequency.
- Applying **Apriori algorithm** using three different libraries:
  - `apriori_python`
  - `apyori`
  - `efficient_apriori`
- Applying **FP-Growth** algorithm from `fpgrowth_py`.
- Comparing execution times of all algorithms.
- Repeating the process on a second dataset (`data.csv`) and comparing results.


## **Technologies Used**
- **Python 3.8+**
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `statsmodels`, `catboost`
  - Statistical Tests: `scipy.stats`
  - Dimensionality Reduction: `umap-learn`, `sklearn.manifold`
  - Association Rules: `apriori_python`, `apyori`, `efficient_apriori`, `fpgrowth_py`
- **Datasets:**
  - California Housing
  - Life Expectancy (WHO)
  - Heart Disease Dataset
  - Customer Personality Analysis
  - Market Basket Optimization
  - COVID-19 ECDC Cases
  - Insurance Dataset


## **Key Outcomes**
- Developed end-to-end data analysis pipelines from data loading to model evaluation.
- Gained hands-on experience with a wide range of machine learning algorithms and statistical methods.
- Learned to visualize data effectively for exploratory and explanatory analysis.
- Compared multiple algorithms for classification, clustering, regression, and association rule mining.
- Understood the importance of preprocessing, feature engineering, and model evaluation.

## **Contact**
**Author:** Chan Minh Hang  
**Email:** [your-email]  
**LinkedIn/GitHub:** [your-profiles]
