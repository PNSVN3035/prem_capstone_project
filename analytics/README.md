# Module 2 — Analytics Pipeline

## Overview

In this module, I worked with the Titanic dataset to build a complete analytics and machine-learning workflow. I started by understanding the raw data, checking missing values, cleaning the dataset, and exploring the relationships between passenger information and survival.

After the exploratory analysis, I used the same dataset to build three classification models for predicting survival. I also compared different class-imbalance techniques, tuned a Random Forest model, built a Linear Regression model to predict fare, and finally saved the best complete classification pipeline so it can be reused on raw passenger data.

The main files used in this module are:

```text
analytics/
├── 01_eda.py
├── 02_modeling.py
├── titanic.csv
├── best_model_pipeline.joblib
├── README.md
└── outputs/
```

---

# Part A — Data Profiling, Cleaning and EDA

## Dataset Loading

I loaded the Titanic dataset using Seaborn:

```python
sns.load_dataset("titanic")
```

The dataset was loaded from Seaborn only once. Immediately after loading it, I saved a local copy:

```text
analytics/titanic.csv
```

The local CSV works as an offline fallback and is also used by the modeling script. This means the modeling stage does not call `sns.load_dataset()` again.

The original dataset has:

```text
891 rows
15 columns
```

---

## Initial Data Profiling

Before cleaning the dataset, I checked:

```python
df.shape
df.info()
df.describe()
```

This helped me understand the number of records, column data types, ranges of the numeric values, and which columns contained missing data.

The columns with missing values were:

| Column      | Missing Values | Missing Percentage |
| ----------- | -------------: | -----------------: |
| age         |            177 |             19.87% |
| embarked    |              2 |              0.22% |
| deck        |            688 |             77.22% |
| embark_town |              2 |              0.22% |

I followed the cleaning rule given in the assignment:

* Below 5% missing → drop the affected rows
* Between 5% and 30% missing → impute the missing values
* Very high missing percentage → make a justified decision to drop the column or treat missing as a separate category

---

## Missing-Value Handling

### Age — 19.87% Missing

The `age` column had 19.87% missing data.

Since this falls between 5% and 30%, I used median imputation.

The median age was:

```text
28 years
```

I preferred the median because it is less affected by unusually high or low ages compared with the mean.

### Embarked — 0.22% Missing

The `embarked` column had only 0.22% missing values.

Since this is below 5%, I removed the affected rows rather than estimating where those passengers boarded.

### Embark Town — 0.22% Missing

`embark_town` also had 0.22% missing values.

The missing records corresponded to the same very small group of passengers, so they were removed under the below-5% rule.

### Deck — 77.22% Missing

The `deck` column had 77.22% missing information.

I decided to remove this column because most passengers did not have a known deck value. Filling more than three-quarters of the column would introduce too much estimated information and could make the analysis misleading.

After cleaning, the EDA dataset contained:

```text
889 rows
14 columns
```

The final check showed:

```text
No missing values remain in the cleaned EDA data.
```

I kept the original `titanic.csv` unchanged and performed the EDA cleaning on a separate DataFrame.

---

# Univariate Analysis

I used histograms and box plots to study `age` and `fare`.

## Age

The age histogram shows that most passengers were young or middle-aged, while fewer passengers were present in the older age groups.

The age box plot also showed some unusually low and high ages. I used the IQR rule to determine the outliers instead of deciding only from the visual appearance of the chart.

## Fare

The fare histogram shows that most passengers paid relatively low fares, while a smaller number of passengers paid much higher amounts.

The fare box plot contains many points beyond the upper whisker. I did not automatically remove these observations because high fares can be genuine passenger records, especially for first-class passengers.

---

# IQR Outlier Analysis

I used:

```text
IQR = Q3 - Q1
```

and defined the boundaries as:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

## Age

For age:

```text
Q1 = 22.00
Q3 = 35.00
IQR = 13.00

Lower Bound = 2.50
Upper Bound = 54.50
```

The number of age outliers was:

```text
65
```

## Fare

For fare:

```text
Q1 = 7.90
Q3 = 31.00
IQR = 23.10

Lower Bound = -26.76
Upper Bound = 65.66
```

The number of fare outliers was:

```text
114
```

Fare therefore contains more IQR outliers than age.

I kept these observations because an outlier is not automatically an incorrect value. In the Titanic dataset, expensive tickets can be valid observations.

---

# Fare Distribution

The fare statistics were:

```text
Mean   = 32.10
Median = 14.45
Mode   = 8.05
```

The ordering is:

```text
Mean > Median > Mode
```

This is consistent with a **right-skewed distribution**.

A relatively small number of passengers paid very high fares. These large values pull the mean upward, while the median and mode remain closer to the fares paid by a larger number of passengers.

---

# Survival Rate by Sex

Since `survived` contains:

```text
0 = Did not survive
1 = Survived
```

the average of this column within a passenger group gives that group's survival rate.

The results were:

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        74.04% |
| Male   |        18.89% |

Female passengers had a much higher survival rate than male passengers.

This was one of the strongest patterns visible in the EDA.

---

# Survival Rate by Passenger Class

The survival rates by class were:

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| First Class     |        62.62% |
| Second Class    |        47.28% |
| Third Class     |        24.24% |

First-class passengers had the highest survival rate, while third-class passengers had the lowest.

This suggests that passenger class was also strongly connected with survival.

---

# Survival Rate by Sex and Passenger Class

I used Boolean masking with the `&` operator to combine sex and passenger class.

The results were:

| Sex    | Class | Survival Rate |
| ------ | ----: | ------------: |
| Female |     1 |        96.74% |
| Female |     2 |        92.11% |
| Female |     3 |        50.00% |
| Male   |     1 |        36.89% |
| Male   |     2 |        15.74% |
| Male   |     3 |        13.54% |

The combination of sex and class gives a clearer story than either feature alone.

First- and second-class female passengers had survival rates above 90%, while male passengers had much lower survival rates across all three classes.

I also demonstrated an OR condition using:

```python
(df["sex"] == "female") | (df["pclass"] == 1)
```

This selects passengers who were female, first class, or both.

---

# Correlation Analysis

The correlation matrix was restricted to exactly the six required columns:

```text
survived
pclass
age
sibsp
parch
fare
```

I intentionally excluded `adult_male` and `alone` because they are derived Boolean features rather than independent measured variables.

The final heatmap therefore contains exactly:

```text
6 × 6
```

Instead of choosing the strongest relationships visually, I ranked all unique off-diagonal pairs by the absolute value of their correlation coefficient.

The two strongest correlations were:

### 1. Passenger Class and Fare

```text
Correlation = -0.548
```

This was the strongest absolute correlation in the six-column matrix.

The negative relationship means that as the numerical passenger-class value increases from first class toward third class, fare generally decreases. This makes practical sense because first-class passengers usually paid higher ticket prices.

### 2. SibSp and Parch

```text
Correlation = 0.415
```

This was the second strongest relationship.

Passengers travelling with siblings or spouses were also somewhat more likely to be travelling with parents or children. This suggests that some passengers were travelling as family groups.

---

# Multivariate Data Story

I created four main charts to build a broader story around survival.

## Chart 1 — Survival by Sex

The first chart shows a major survival difference between female and male passengers.

Female passengers had a much higher survival rate. This indicates that sex was strongly associated with the chance of survival and would likely be useful in a predictive model.

## Chart 2 — Survival by Passenger Class and Sex

The second chart considers sex and passenger class together.

Female passengers generally had higher survival rates across all classes, but passenger class also mattered. First-class passengers tended to have better outcomes, showing that survival was connected to more than one passenger characteristic.

## Chart 3 — Age, Sex and Survival

The third chart compares age distributions across survival status while separating passengers by sex.

There is considerable overlap between the age groups, so age alone does not clearly separate survivors from non-survivors. Age becomes more useful when it is considered together with other passenger characteristics.

## Chart 4 — Fare, Passenger Class and Survival

The fourth chart compares fare, passenger class and survival.

First-class passengers generally paid higher fares, showing that fare and passenger class contain related socioeconomic information. There are also differences between survivors and non-survivors, although fare alone does not fully explain survival.

---

# Standardization Check

As an exploratory check, I standardized `age` and `fare` using the z-score formula:

```text
z = (x - mean) / standard deviation
```

Before standardization, age and fare were measured on different scales.

After standardization, both transformed columns had approximately:

```text
Mean = 0
Standard Deviation = 1
```

This confirmed that the z-score transformation worked correctly.

This transformation was used only for EDA.

I did not pass these EDA-standardized columns into the machine-learning models. The actual modeling pipeline uses its own `StandardScaler`, fitted only on the training data, which avoids data leakage.

---

# Part B — Predictive Modeling, Regression

## Class Balance

The original survival distribution was:

| Class           | Count | Percentage |
| --------------- | ----: | ---------: |
| Did Not Survive |   549 |     61.62% |
| Survived        |   342 |     38.38% |

The classes are not equally distributed.

I therefore used a **stratified train/test split** so that the survival proportions remained similar in both datasets.

The split produced:

```text
Training Rows = 712
Testing Rows = 179
```

Training target distribution:

```text
Not Survived = 61.66%
Survived     = 38.34%
```

Testing target distribution:

```text
Not Survived = 61.45%
Survived     = 38.55%
```

The percentages are very similar, confirming that stratification worked as intended.

---

# Features Used for Classification

Numeric features:

```text
pclass
age
sibsp
parch
fare
```

Categorical features:

```text
sex
embarked
```

Target:

```text
survived
```

---

# Preprocessing and Data Leakage Prevention

The train/test split was performed **before fitting any preprocessing step**.

For numeric columns, the pipeline performs:

```text
Median Imputation
        ↓
StandardScaler
```

For categorical columns:

```text
Most-Frequent Imputation
        ↓
OneHotEncoder
```

These preprocessing operations are handled using `ColumnTransformer` and `Pipeline`.

When:

```python
pipeline.fit(X_train, y_train)
```

is called, the imputer, encoder and scaler learn only from the training data.

The test data is never used to fit or refit preprocessing. It is only transformed using the parameters learned from training data.

This prevents test-set information from leaking into the training process.

---

# Models Trained

I trained three classification models using the same training and testing split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

I also visualized the Decision Tree using `plot_tree()` with the transformed feature names and survival class labels.

---

# Classification Model Results

| Model               |   Accuracy |  Precision |     Recall |         F1 |        AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.8045 |     0.7931 |     0.6667 |     0.7244 | **0.8437** |
| Decision Tree       |     0.7933 | **0.8636** |     0.5507 |     0.6726 |     0.8292 |
| Random Forest       | **0.8212** |     0.8136 | **0.6957** | **0.7500** |     0.8300 |

No single classifier produced the best value for every metric.

Random Forest had the highest accuracy, recall and F1 score.

Decision Tree had the highest precision, but its lower recall means that it missed more passengers who actually survived.

Logistic Regression had the highest ROC-AUC, showing strong ranking ability across different classification thresholds.

---

# Confusion Matrices

## Logistic Regression

```text
[[98, 12],
 [23, 46]]
```

This means:

```text
True Negatives  = 98
False Positives = 12
False Negatives = 23
True Positives  = 46
```

## Decision Tree

```text
[[104, 6],
 [31, 38]]
```

The Decision Tree produced only six false positives, which helps explain its high precision of 0.8636.

However, it missed 31 actual survivors, resulting in lower recall.

## Random Forest

```text
[[99, 11],
 [21, 48]]
```

Random Forest correctly identified:

```text
99 non-survivors
48 survivors
```

It produced:

```text
11 false positives
21 false negatives
```

Compared with the Decision Tree, Random Forest identified more actual survivors while still maintaining good precision.

---

# ROC-AUC Analysis

The AUC results were:

```text
Logistic Regression = 0.8437
Decision Tree        = 0.8292
Random Forest        = 0.8300
```

Logistic Regression had the highest AUC.

Random Forest, however, had the strongest accuracy, recall and F1 result.

This demonstrates why I did not select a model using only one metric.

---

# Class-Imbalance Comparison

The training target consisted of:

```text
Not Survived = 61.66%
Survived     = 38.34%
```

I compared Logistic Regression using three approaches:

1. Baseline with no special handling
2. `class_weight="balanced"`
3. SMOTE

SMOTE was applied only to the training stage of the pipeline. The test data was never oversampled.

The results were:

| Strategy              |  Precision |     Recall |         F1 |
| --------------------- | ---------: | ---------: | ---------: |
| Baseline              | **0.7931** |     0.6667 |     0.7244 |
| Class Weight Balanced |     0.7297 | **0.7826** |     0.7552 |
| SMOTE                 |     0.7397 | **0.7826** | **0.7606** |

Both imbalance-handling approaches improved recall.

The baseline model had higher precision, but its recall was lower.

SMOTE increased recall from:

```text
0.6667 → 0.7826
```

and produced the highest F1 score:

```text
0.7606
```

For this imbalance experiment, **SMOTE gave the best overall precision-recall balance**.

This does not mean SMOTE automatically replaces Random Forest as the final classifier. The imbalance experiment uses Logistic Regression specifically to compare imbalance-handling methods under the same model.

---

# Random Forest Hyperparameter Tuning

I used `GridSearchCV` with 5-fold cross-validation to tune:

```text
n_estimators
max_depth
max_features
```

The best parameter combination was:

```text
n_estimators = 100
max_depth = 8
max_features = sqrt
```

The best cross-validation F1 score was:

```text
0.7484
```

I created the Random Forest with:

```python
oob_score=True
```

so I could also report the required out-of-bag score.

The OOB score was:

```text
0.8188
```

---

# Tuned Random Forest

The tuned Random Forest produced:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.8045 |
| Precision | 0.8036 |
| Recall    | 0.6522 |
| F1        | 0.7200 |
| AUC       | 0.8401 |

The tuned model did not outperform the original Random Forest on held-out F1.

Original Random Forest:

```text
F1 = 0.7500
```

Tuned Random Forest:

```text
F1 = 0.7200
```

This is possible because GridSearchCV selects the parameter combination that performs best on cross-validation folds within the training data. That does not guarantee that it will achieve a better result on one separate test set.

For this reason, I did not automatically choose the tuned model just because hyperparameter tuning had been performed.

---

## Fare Prediction

For the regression task, I changed the target from survival to:

```text
fare
```

The predictors were:

```text
pclass
age
sibsp
parch
sex
embarked
```

I used multivariate Linear Regression with preprocessing handled inside a pipeline.

---

# Regression Results

| Metric      |  Result |
| ----------- | ------: |
| MAE         | 20.8094 |
| RMSE        | 30.4731 |
| R²          |  0.3999 |
| Adjusted R² |  0.3679 |

## MAE

The MAE of **20.8094** means that the model's predicted fare differs from the actual fare by approximately 20.81 fare units on average when using absolute error.

## RMSE

The RMSE was:

```text
30.4731
```

It is higher than MAE because RMSE gives greater weight to large prediction errors.

The difference between MAE and RMSE suggests that some passengers had relatively large fare-prediction errors.

## R²

The R² score was:

```text
0.3999
```

This means that the model explains approximately 40% of the variation in passenger fares in the test data.

There is therefore still a substantial amount of variation that the selected features do not explain.

## Adjusted R²

Adjusted R² was:

```text
0.3679
```

It is lower than R² because it takes the number of predictors into account.

The difference shows that adding predictors does not automatically mean every feature contributes equally useful explanatory information.

---

# Residual Analysis and Heteroscedasticity

Residuals were calculated as:

```text
Residual = Actual Fare - Predicted Fare
```

The residual plot showed that the spread of the residuals becomes noticeably wider as predicted fares increase instead of remaining approximately constant.

This indicates **heteroscedasticity**.

In other words, the Linear Regression model makes relatively smaller and more consistent errors for lower predicted fares, while the error spread becomes much larger for higher predicted fares.

This means the constant-error-variance assumption of ordinary Linear Regression is not fully satisfied for this fare model.

---

# Final Model Comparison

Classification and regression solve different problems, so their metrics are kept as separate groups.

| Model Type     | Model               | Accuracy | Precision | Recall |     F1 |    AUC |     MAE |    RMSE |     R² | Adjusted R² |
| -------------- | ------------------- | -------: | --------: | -----: | -----: | -----: | ------: | ------: | -----: | ----------: |
| Classification | Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |       — |       — |      — |           — |
| Classification | Decision Tree       |   0.7933 |    0.8636 | 0.5507 | 0.6726 | 0.8292 |       — |       — |      — |           — |
| Classification | Random Forest       |   0.8212 |    0.8136 | 0.6957 | 0.7500 | 0.8300 |       — |       — |      — |           — |
| Regression     | Linear Regression   |        — |         — |      — |      — |      — | 20.8094 | 30.4731 | 0.3999 |      0.3679 |

Classification metrics such as accuracy and F1 cannot be directly compared with regression metrics such as RMSE and R² because they measure completely different types of prediction performance.

---

# Final Model Recommendation

For the main survival-classification problem, I would choose **Random Forest** from the three required classifiers.

It achieved the highest accuracy of **0.8212**, the highest recall of **0.6957**, and the highest F1 score of **0.7500**. Its precision was also strong at **0.8136**, giving it a good balance between correctly identifying survivors and avoiding incorrect survivor predictions.

Logistic Regression remains a strong alternative because it achieved the highest AUC of **0.8437**, but its F1 score was lower at **0.7244**. The Decision Tree had the highest precision but considerably lower recall. Based on the overall balance of the required metrics, Random Forest is my preferred classifier.

---

# Saving the Best Pipeline

I saved the selected Random Forest as a complete fitted pipeline:

```text
analytics/best_model_pipeline.joblib
```

I did not save only the Random Forest estimator.

The saved object contains:

```text
Raw Passenger Data
        ↓
Numeric Imputation
        ↓
Standard Scaling
        ↓
Categorical Imputation
        ↓
One-Hot Encoding
        ↓
Random Forest
        ↓
Prediction
```

This makes the saved model usable directly with raw passenger data.

---

# Reloading the Pipeline

I reloaded the saved object using:

```python
joblib.load()
```

I then tested it using raw records from the test dataset.

One of the test records even contained a missing `age` value, while categorical features still contained raw values such as:

```text
male
female
S
C
Q
```

The reloaded pipeline successfully performed the required preprocessing itself.

The predictions were:

```text
[0 0 0 0 1]
```

I compared these predictions with the predictions produced by the original pipeline before it was saved.

The result was:

```text
Original and Reloaded Predictions Match:
True
```

This confirms that the saved pipeline can be loaded again and used end-to-end on raw input without manually repeating the preprocessing steps.

---

# Final Findings

The exploratory analysis showed that sex and passenger class had clear relationships with survival. Female passengers had much higher survival rates than male passengers, and first-class passengers generally had better survival outcomes than passengers in lower classes.

Among the three required classifiers, Random Forest provided the best overall balance. It achieved **82.12% accuracy** and an F1 score of **0.7500**.

The imbalance experiment also showed the importance of looking beyond accuracy. SMOTE increased Logistic Regression recall from **0.6667 to 0.7826** and produced an F1 score of **0.7606**.

Hyperparameter tuning did not automatically improve test performance. The tuned Random Forest had an F1 score of **0.7200**, compared with **0.7500** for the original Random Forest.

For fare prediction, Linear Regression explained around 40% of the variation in fares. The residual analysis also showed heteroscedasticity, which suggests that prediction error becomes less consistent for passengers with higher predicted fares.

Overall, this module showed the complete process from understanding and cleaning raw data to exploring relationships, building models, evaluating them carefully, handling class imbalance, tuning parameters, and saving a reusable machine-learning pipeline.

---

# How to Run Module 2

From the project root, activate the virtual environment:

```bash
source venv/bin/activate
```

Install all project dependencies:

```bash
pip install -r requirements.txt
```

Run the EDA pipeline first:

```bash
python3 analytics/01_eda.py
```

Then run the modeling pipeline:

```bash
python3 analytics/02_modeling.py
```

The EDA script loads or reads the Titanic dataset, performs profiling and cleaning, calculates the required statistics, and generates the EDA charts.

The modeling script performs the stratified split, preprocessing, classification, imbalance comparison, hyperparameter tuning, regression analysis, model saving, and model reload verification.

---

# Main Outputs

The main outputs from Module 2 are:

```text
analytics/
├── 01_eda.py
├── 02_modeling.py
├── titanic.csv
├── best_model_pipeline.joblib
├── README.md
│
└── outputs/
    ├── missing_values_report.csv
    ├── eda_summary.csv
    ├── sex_class_survival_rates.csv
    ├── correlation_matrix.csv
    ├── top_two_correlations.csv
    ├── classification_model_comparison.csv
    ├── class_balance.csv
    ├── imbalance_comparison.csv
    ├── random_forest_grid_search.csv
    ├── regression_metrics.csv
    ├── final_model_comparison.csv
    │
    └── charts/
        ├── age_histogram.png
        ├── age_boxplot.png
        ├── fare_histogram.png
        ├── fare_boxplot.png
        ├── correlation_heatmap.png
        ├── survival_by_sex.png
        ├── survival_by_class_sex.png
        ├── age_survival_sex.png
        ├── fare_class_survival.png
        ├── logistic_regression_confusion_matrix.png
        ├── decision_tree_confusion_matrix.png
        ├── random_forest_confusion_matrix.png
        ├── classification_roc_curves.png
        ├── decision_tree.png
        └── fare_regression_residual_plot.png
```

---

## Conclusion

I kept the EDA and modeling stages separate so that each part of the workflow has a clear responsibility. The EDA stage helps understand the data and identify useful patterns, while the modeling stage performs its own preprocessing after the train/test split to avoid leakage.

I also kept classification and regression results separate because they answer different questions and use different evaluation metrics.

The final result is a reproducible analytics workflow that starts with raw Titanic data and ends with an evaluated and reusable machine-learning pipeline.
