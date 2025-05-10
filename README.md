GROUP 12

**Procedures for Breast Cancer Data Analysis**

This report outlines the procedure followed to analyze the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses, with the goal of classifying tumors as malignant (M) or benign (B). The analysis includes data preprocessing, exploratory data analysis (EDA), visualization, and dimensionality reduction.

**1.DATA PREPROCESSING**

**1.1 Loading and Initial Inspection**

The dataset was loaded using pandas.read_csv().

Initial inspection included:

Checking dataset dimensions (df.shape → 569 samples, 32 features).

Displaying the first few rows (df.head()).

Verifying data types and null values (df.info(), df.isnull().sum()).

Checking for duplicates (df.duplicated().sum()).

Findings:

No missing values or duplicates were found.

Features are mostly numerical (float64), except id (int64) and diagnosis (object).

**1.2 Encoding Categorical Data**

The target variable diagnosis (M/B) was encoded numerically using LabelEncoder:

M (Malignant) → 1

B (Benign) → 0

**1.3 Feature Scaling**

Numerical features (excluding id and diagnosis) were standardized using StandardScaler to ensure zero mean and unit variance.

**1.4 Outlier Detection**

Outliers were identified but were kept since they represented aggresive forms of cancer however if we were to remove the outliers we would use the IQR method  

Lower bound = Q1 – 1.5 × IQR

Upper bound = Q3 + 1.5 × IQR

After removal, the dataset reduced from 569 to 277 samples.

**2.EXPLORATORY DATA ANALYSIS(EDA)**


**2.1 Summary Statistics**
Descriptive statistics (df.describe()) revealed:

Mean values (e.g., radius_mean = 14.13, area_mean = 654.89).

Standard deviations, min/max values, and quartiles.

**2.2 Class Distribution**
The dataset was imbalanced:

Benign (B): 357 cases (62.74%)

Malignant (M): 212 cases (37.26%)

**2.3 Correlation Analysis**
A heatmap of the correlation matrix showed:

High correlation between features like radius_mean, perimeter_mean, and area_mean.

Some features (e.g., concave points_mean) strongly correlated with diagnosis.

**2.4 Feature Importance**

Correlation with diagnosis (diagnosis_numeric) was computed:

Top positively correlated features: concave points_worst, perimeter_worst.

Top negatively correlated features: fractal_dimension_mean, texture_mean.

**3.DATA VISUALIZATION**

**3.1 Count Plot**

A bar plot showed the distribution of malignant vs. benign cases.

**3.2 Histograms & KDE Plots**

Histograms with Kernel Density Estimation (KDE) were plotted for mean features (e.g., radius_mean, texture_mean).

Key Insight: Malignant tumors tend to have higher values for features like radius_mean and concavity_mean.

**3.3 Box Plots**

Box plots were generated to compare feature distributions:

Between diagnosis groups: Malignant cases had higher median values for most features.

Overall distributions: Some features (e.g., area_mean) had significant outliers.

**3.4 Pair Plot**

A pair plot of selected features (radius_mean, texture_mean, etc.) showed clear separation between malignant and benign cases.

**4.Dimensionality Reduction (PCA)**

**4.1 Principal Component Analysis (PCA)**

Features were standardized and reduced to 2 principal components.

Explained Variance:

PC1: ~44% of variance.

PC2: ~19% of variance.

A scatter plot of PC1 vs. PC2 showed good separation between malignant and benign cases.

**5.1 Cumulative Explained Variance**

A scatter plot indicated that ~95% variance was explained by the first 10 components.

**6.Key Findings**

Malignant tumors generally have higher values for features like radius_mean, concavity_mean, and area_mean.

Feature correlations suggest redundancy (e.g., radius_mean and perimeter_mean are highly correlated).

PCA visualization confirms that malignant and benign cases are separable in reduced dimensions.

Outliers were present in many features, affecting model robustness.

**7.Recommendations**

Feature Selection: Remove highly correlated features to reduce multicollinearity.

Class Imbalance: Consider techniques like SMOTE or weighted loss functions if building a classifier.

Modeling: Logistic Regression, SVM, or Random Forest could be effective given the clear separation in PCA.

**8.Conclusion**

This analysis provided insights into the dataset through preprocessing, visualization, and statistical summaries. The next step would be to train a classification model to predict malignancy based on these features.

**BREAST CANCER CLASSIFICATION WITH MACHINE LEARNING** 
 
**1. Data Preparation & Preprocessing**

**1.1 Loading & Initial Inspection**

Dataset: 569 samples, 30 features + diagnosis (M/B)

Checked for missing values & duplicates (none found)

Encoded target: M → 1, B → 0

**1.2 Feature Engineering**

Dropped ID column (non-predictive)

Standardized features using StandardScaler() (critical for KNN)

Train-test split (60-40) with stratification to preserve class balance

**2. Model Implementation**
   
**2.1 Random Forest Classifier**

Initial Model:

500 trees, max_depth=15

Accuracy: 95.6%, Recall: 89% (malignant)

Hyperparameter Tuning:

RandomizedSearchCV optimized for recall

Best params: n_estimators=200, class_weight='balanced'

Improved Recall: 94% (malignant)

**2.2 K-Nearest Neighbors**

Standardized features required

Default k=5 neighbors

Accuracy: 96.1%, Recall: 89% (malignant)

Cross-Validation AUC: 0.968

Winner: Random Forest (better at catching cancer cases)

**3.2 Key Visualizations**

Confusion Matrices: Both models showed excellent benign detection (99%+)

Feature Importance: worst concave points most predictive in RF

ROC Curves: Both models achieved perfect AUC=1.0 on test set

**4.Critical Insights**
   
Class Imbalance Matters:

Untuned RF missed 11% malignant cases

Class weighting improved malignant recall by 5%

Feature Scaling is Crucial:

KNN performance dropped 3% without standardization

Model Interpretability Tradeoff:

RF provides feature importance

KNN offers simpler implementation

End of Report.


**Conclusion:**
The analysis demonstrated the effectiveness of machine learning techniques in predicting breast cancer diagnosis.


