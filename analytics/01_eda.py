from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#file paths
BASE_DIR = Path(__file__).resolve().parent

TITANIC_FILE = BASE_DIR / "titanic.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#load titanic dataset

if TITANIC_FILE.exists():

    print("Loading Titanic dataset from saved local CSV...")

    df = pd.read_csv(TITANIC_FILE)

else:

    print("Loading Titanic dataset from Seaborn...")

    df = sns.load_dataset("titanic")

    df.to_csv(
        TITANIC_FILE,
        index=False
    )

    print(f"Titanic dataset saved to: {TITANIC_FILE}")

#profiling

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

#missing value analysis

print("\n" + "=" * 60)
print("MISSING VALUE REPORT")
print("=" * 60)


missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)


missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

#missing values
missing_report = missing_report[
    missing_report["missing_count"] > 0
]

print(missing_report)

#save the missing value report
missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv"
)

#make an EDA cleaning copy
eda_df = df.copy()

#Clean AGE
age_missing_percentage = (
    eda_df["age"].isnull().mean() * 100
)


print(
    f"\nAge missing percentage: "
    f"{age_missing_percentage:.2f}%"
)

#rule: 5%–30% missing -> impute

if 5 <= age_missing_percentage <= 30:

    age_median = eda_df["age"].median()

    eda_df["age"] = (
        eda_df["age"]
        .fillna(age_median)
    )

    print(
        "Age strategy: Median imputation "
        "because missing percentage is between 5% and 30%."
    )

    print(
        f"Age median used: {age_median:.2f}"
    )

#clean EMBARKED

embarked_missing_percentage = (
    eda_df["embarked"].isnull().mean() * 100
)


print(
    f"\nEmbarked missing percentage: "
    f"{embarked_missing_percentage:.2f}%"
)

#rule: <5% missing -> drop affected rows

if 0 < embarked_missing_percentage < 5:

    eda_df = eda_df.dropna(
        subset=["embarked"]
    ).copy()

    print(
        "Embarked strategy: Rows with missing embarked "
        "values were dropped because missing percentage "
        "is below 5%."
    )

#clean EMBARK_TOWN

embark_town_missing_percentage = (
    eda_df["embark_town"].isnull().mean() * 100
)


print(
    f"\nEmbark town missing percentage: "
    f"{embark_town_missing_percentage:.2f}%"
)

#rule: <5% missing -> drop affected rows
if 0 < embark_town_missing_percentage < 5:

    eda_df = eda_df.dropna(
        subset=["embark_town"]
    ).copy()

    print(
        "Embark town: Rows with missing values "
        "were dropped because missing percentage "
        "is below 5%."
    )

#clean DECK

deck_missing_percentage = (
    eda_df["deck"].isnull().mean() * 100
)


print(
    f"\nDeck missing percentage: "
    f"{deck_missing_percentage:.2f}%"
)

#very high missing -> drop column
if deck_missing_percentage > 30:

    eda_df = eda_df.drop(
        columns=["deck"]
    )

    print(
        "Deck strategy: Column dropped because the missing "
        "percentage is very high. Imputing most of this "
        "column could create unreliable information."
    )

#verify cleaning

print("\n" + "=" * 60)
print("MISSING VALUES AFTER EDA CLEANING")
print("=" * 60)

remaining_missing = eda_df.isnull().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
]

if remaining_missing.empty:

    print("No missing values remain in the cleaned EDA data.")

else:

    print(remaining_missing)

#final cleaned shape

print("\n" + "=" * 60)
print("CLEANED EDA DATASET SHAPE")
print("=" * 60)

print(eda_df.shape)

#eda analysis

#create folder

CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

#univariate analysis
print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

#age histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    data=eda_df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "age_histogram.png"
)

plt.show()

#age box plot
plt.figure(figsize=(8, 4))

sns.boxplot(
    data=eda_df,
    x="age"
)

plt.title("Box Plot of Passenger Age")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "age_boxplot.png"
)

plt.show()

#fare histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    data=eda_df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "fare_histogram.png"
)

plt.show()

#fare box plot
plt.figure(figsize=(8, 4))

sns.boxplot(
    data=eda_df,
    x="fare"
)

plt.title("Box Plot of Passenger Fare")
plt.xlabel("Fare")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "fare_boxplot.png"
)

plt.show()

#IQR outliers
def find_iqr_outliers(data, column):

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = data[
        (data[column] < lower_bound)
        |
        (data[column] > upper_bound)
    ]

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        outliers
    ) 

#age outliers

(
    age_q1,
    age_q3,
    age_iqr,
    age_lower,
    age_upper,
    age_outliers
) = find_iqr_outliers(
    eda_df,
    "age"
)


print("\nAGE OUTLIER REPORT")

print(f"Q1: {age_q1:.2f}")
print(f"Q3: {age_q3:.2f}")
print(f"IQR: {age_iqr:.2f}")
print(f"Lower Bound: {age_lower:.2f}")
print(f"Upper Bound: {age_upper:.2f}")

print(
    f"Number of Age Outliers: "
    f"{len(age_outliers)}"
)

#fare outliers

(
    fare_q1,
    fare_q3,
    fare_iqr,
    fare_lower,
    fare_upper,
    fare_outliers
) = find_iqr_outliers(
    eda_df,
    "fare"
)


print("\nFARE OUTLIER REPORT")

print(f"Q1: {fare_q1:.2f}")
print(f"Q3: {fare_q3:.2f}")
print(f"IQR: {fare_iqr:.2f}")
print(f"Lower Bound: {fare_lower:.2f}")
print(f"Upper Bound: {fare_upper:.2f}")

print(
    f"Number of Fare Outliers: "
    f"{len(fare_outliers)}"
)

#fare (mean, median & mode)

fare_mean = eda_df["fare"].mean()
fare_median = eda_df["fare"].median()

fare_modes = eda_df["fare"].mode()
fare_mode = fare_modes.iloc[0]


print("\n" + "=" * 60)
print("FARE STATISTICS")
print("=" * 60)

print(f"Fare Mean: {fare_mean:.2f}")
print(f"Fare Median: {fare_median:.2f}")
print(f"Fare Mode: {fare_mode:.2f}")


if (
    fare_mean > fare_median
    and fare_median >= fare_mode
):

    fare_skew = "right-skewed"

elif (
    fare_mean < fare_median
    and fare_median <= fare_mode
):

    fare_skew = "left-skewed"

else:

    fare_skew = "not perfectly determined by ordering alone"


print(
    f"Fare distribution conclusion: "
    f"{fare_skew}"
)

#bivariate analysis

#survival rate by sex
print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX")
print("=" * 60)


male_mask = (
    eda_df["sex"] == "male"
)

female_mask = (
    eda_df["sex"] == "female"
)


male_survival_rate = (
    eda_df.loc[
        male_mask,
        "survived"
    ].mean() * 100
)


female_survival_rate = (
    eda_df.loc[
        female_mask,
        "survived"
    ].mean() * 100
)


print(
    f"Male Survival Rate: "
    f"{male_survival_rate:.2f}%"
)

print(
    f"Female Survival Rate: "
    f"{female_survival_rate:.2f}%"
)

#survival rate by passenger class

print("\n" + "=" * 60)
print("SURVIVAL RATE BY PASSENGER CLASS")
print("=" * 60)


class_survival_rates = {}


for passenger_class in [1, 2, 3]:

    class_mask = (
        eda_df["pclass"] == passenger_class
    )

    survival_rate = (
        eda_df.loc[
            class_mask,
            "survived"
        ].mean() * 100
    )

    class_survival_rates[
        passenger_class
    ] = survival_rate

    print(
        f"Class {passenger_class}: "
        f"{survival_rate:.2f}%"
    )

#survival rate by sex and passenger class  

#boolean masking with &

print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX AND CLASS")
print("=" * 60)


sex_class_results = []


for sex in ["female", "male"]:

    for passenger_class in [1, 2, 3]:

        mask = (
            (eda_df["sex"] == sex)
            &
            (
                eda_df["pclass"]
                == passenger_class
            )
        )

        survival_rate = (
            eda_df.loc[
                mask,
                "survived"
            ].mean() * 100
        )

        sex_class_results.append(
            {
                "sex": sex,
                "pclass": passenger_class,
                "survival_rate":
                    round(
                        survival_rate,
                        2
                    )
            }
        )

        print(
            f"{sex.title()} - "
            f"Class {passenger_class}: "
            f"{survival_rate:.2f}%"
        )


sex_class_df = pd.DataFrame(
    sex_class_results
)

#boolean masking with |
female_or_first_class_mask = (
    (eda_df["sex"] == "female")
    |
    (eda_df["pclass"] == 1)
)


female_or_first_class_rate = (
    eda_df.loc[
        female_or_first_class_mask,
        "survived"
    ].mean() * 100
)


print(
    "\nFemale OR First-Class "
    "Survival Rate: "
    f"{female_or_first_class_rate:.2f}%"
)

#correlation matrix exactly six columns required
print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)


correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


correlation_matrix = (
    eda_df[
        correlation_columns
    ].corr()
)


print(correlation_matrix)

#correlation heatmap
plt.figure(
    figsize=(9, 6)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix of Selected Titanic Features"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "correlation_heatmap.png"
)

plt.show()

#find two strongest correlations

correlation_pairs = []

columns = correlation_matrix.columns


for i in range(
    len(columns)
):

    for j in range(
        i + 1,
        len(columns)
    ):

        feature_1 = columns[i]
        feature_2 = columns[j]

        correlation_value = (
            correlation_matrix.loc[
                feature_1,
                feature_2
            ]
        )

        correlation_pairs.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation":
                    correlation_value,
                "absolute_correlation":
                    abs(
                        correlation_value
                    )
            }
        )


correlation_pairs_df = pd.DataFrame(
    correlation_pairs
)


correlation_pairs_df = (
    correlation_pairs_df
    .sort_values(
        by="absolute_correlation",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


top_two_correlations = (
    correlation_pairs_df.head(2)
)


print("\nTWO STRONGEST CORRELATIONS")

print(
    top_two_correlations[
        [
            "feature_1",
            "feature_2",
            "correlation"
        ]
    ]
)

#multivariate data story

#4 required charts

print("\n" + "=" * 60)
print("CREATING MULTIVARIATE CHARTS")
print("=" * 60)

#chart 1

#survival by sex

plt.figure(
    figsize=(7, 5)
)

sns.barplot(
    data=eda_df,
    x="sex",
    y="survived"
)

plt.title(
    "Survival Rate by Sex"
)

plt.xlabel("Sex")
plt.ylabel(
    "Average Survival Rate"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "survival_by_sex.png"
)

plt.show()

#chart 2

#survival by class and sex

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=eda_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.xlabel(
    "Passenger Class"
)

plt.ylabel(
    "Average Survival Rate"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "survival_by_class_sex.png"
)

plt.show()


#chart 3

#age, survival and sex

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=eda_df,
    x="survived",
    y="age",
    hue="sex"
)

plt.title(
    "Age Distribution by Survival Status and Sex"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "age_survival_sex.png"
)

plt.show()

#chart 4

#fare, class and survival
plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=eda_df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare Distribution by Passenger Class and Survival"
)

plt.xlabel(
    "Passenger Class"
)

plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "fare_class_survival.png"
)

plt.show()


#standardization check

print("\n" + "=" * 60)
print("STANDARDIZATION CHECK")
print("=" * 60)


before_standardization = (
    eda_df[
        [
            "age",
            "fare"
        ]
    ]
    .agg(
        [
            "mean",
            "std"
        ]
    )
)

print(
    "\nBefore Standardization:"
)

print(
    before_standardization
)


#z-score 

eda_df["age_z"] = (
    (
        eda_df["age"]
        - eda_df["age"].mean()
    )
    /
    eda_df["age"].std()
)


eda_df["fare_z"] = (
    (
        eda_df["fare"]
        - eda_df["fare"].mean()
    )
    /
    eda_df["fare"].std()
)


after_standardization = (
    eda_df[
        [
            "age_z",
            "fare_z"
        ]
    ]
    .agg(
        [
            "mean",
            "std"
        ]
    )
)

print(
    "\nAfter Standardization:"
)

print(
    after_standardization
)


eda_summary = pd.DataFrame(
    {
        "metric": [
            "age_outlier_count",
            "fare_outlier_count",
            "fare_mean",
            "fare_median",
            "fare_mode",
            "male_survival_rate",
            "female_survival_rate",
            "class_1_survival_rate",
            "class_2_survival_rate",
            "class_3_survival_rate",
            "female_or_first_class_rate"
        ],

        "value": [
            len(age_outliers),
            len(fare_outliers),
            round(
                fare_mean,
                2
            ),
            round(
                fare_median,
                2
            ),
            round(
                fare_mode,
                2
            ),
            round(
                male_survival_rate,
                2
            ),
            round(
                female_survival_rate,
                2
            ),
            round(
                class_survival_rates[1],
                2
            ),
            round(
                class_survival_rates[2],
                2
            ),
            round(
                class_survival_rates[3],
                2
            ),
            round(
                female_or_first_class_rate,
                2
            )
        ]
    }
)


eda_summary.to_csv(
    OUTPUT_DIR
    / "eda_summary.csv",
    index=False
)


sex_class_df.to_csv(
    OUTPUT_DIR
    / "sex_class_survival_rates.csv",
    index=False
)


correlation_matrix.to_csv(
    OUTPUT_DIR
    / "correlation_matrix.csv"
)


top_two_correlations.to_csv(
    OUTPUT_DIR
    / "top_two_correlations.csv",
    index=False
)

print(
    "\nEDA output files saved successfully."
)