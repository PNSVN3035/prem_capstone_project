from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent

TITANIC_FILE = BASE_DIR / "titanic.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

#load saved dataset
if not TITANIC_FILE.exists():

    raise FileNotFoundError(
        "titanic.csv was not found. "
        "Run 01_eda.py first."
    )


df = pd.read_csv(
    TITANIC_FILE
)


print("\n" + "=" * 60)
print("MODELING DATASET")
print("=" * 60)

print(
    f"Shape: {df.shape}"
)

#class balance

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)


class_counts = (
    df["survived"]
    .value_counts()
    .sort_index()
)


class_percentages = (
    df["survived"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print(
    "Class Counts:"
)

print(
    class_counts
)


print(
    "\nClass Percentages:"
)

print(
    class_percentages.round(2)
)


print(
    "\n0 = Did not survive"
)

print(
    "1 = Survived"
)

# Numeric features
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


# Categorical features required by assignment
categorical_features = [
    "sex",
    "embarked"
]


feature_columns = (
    numeric_features
    + categorical_features
)


X = df[
    feature_columns
].copy()


y = df[
    "survived"
].copy()


print(
    "\nFeatures used:"
)

print(
    feature_columns
)

#train-test split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


print(
    "\nTraining target percentage:"
)

print(
    (
        y_train
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)
)


print(
    "\nTesting target percentage:"
)

print(
    (
        y_test
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)
)

#numeric preprocessing:

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# categorical preprocessing:

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

#logistic regression
logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


logistic_pipeline.fit(
    X_train,
    y_train
)


print(
    "\nLogistic Regression training completed."
)

#decision tree
decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=4,
                random_state=42
            )
        )
    ]
)


decision_tree_pipeline.fit(
    X_train,
    y_train
)


print(
    "Decision Tree training completed."
)

#random forest
random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


random_forest_pipeline.fit(
    X_train,
    y_train
)


print(
    "Random Forest training completed."
)

#model evaluation
def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test
):

    predictions = pipeline.predict(
        X_test
    )


    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    auc_score = roc_auc_score(
        y_test,
        probabilities
    )


    print(
        "\n" + "=" * 60
    )

    print(
        model_name.upper()
    )

    print(
        "=" * 60
    )


    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC AUC:   {auc_score:.4f}"
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "predictions": predictions,
        "probabilities": probabilities
    }

#evaluate all models
logistic_results = evaluate_model(
    "Logistic Regression",
    logistic_pipeline,
    X_test,
    y_test
)


tree_results = evaluate_model(
    "Decision Tree",
    decision_tree_pipeline,
    X_test,
    y_test
)


forest_results = evaluate_model(
    "Random Forest",
    random_forest_pipeline,
    X_test,
    y_test
)

#comparison table
comparison_table = pd.DataFrame(
    [
        {
            "Model":
                logistic_results["model"],

            "Accuracy":
                logistic_results["accuracy"],

            "Precision":
                logistic_results["precision"],

            "Recall":
                logistic_results["recall"],

            "F1":
                logistic_results["f1"],

            "AUC":
                logistic_results["auc"]
        },

        {
            "Model":
                tree_results["model"],

            "Accuracy":
                tree_results["accuracy"],

            "Precision":
                tree_results["precision"],

            "Recall":
                tree_results["recall"],

            "F1":
                tree_results["f1"],

            "AUC":
                tree_results["auc"]
        },

        {
            "Model":
                forest_results["model"],

            "Accuracy":
                forest_results["accuracy"],

            "Precision":
                forest_results["precision"],

            "Recall":
                forest_results["recall"],

            "F1":
                forest_results["f1"],

            "AUC":
                forest_results["auc"]
        }
    ]
)


numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]


comparison_table[
    numeric_columns
] = (
    comparison_table[
        numeric_columns
    ]
    .round(4)
)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison_table.to_string(
        index=False
    )
)


comparison_table.to_csv(
    OUTPUT_DIR
    / "classification_model_comparison.csv",
    index=False
)

#confusion matrix
models = {
    "Logistic Regression":
        logistic_pipeline,

    "Decision Tree":
        decision_tree_pipeline,

    "Random Forest":
        random_forest_pipeline
}


for model_name, model in models.items():

    predictions = model.predict(
        X_test
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Survived",
            "Survived"
        ]
    )


    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_confusion_matrix.png"
    )


    plt.savefig(
        CHART_DIR / filename
    )

    plt.show()

#roc curves
plt.figure(
    figsize=(8, 6)
)


for model_name, model in models.items():

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )


    model_auc = auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC = {model_auc:.3f})"
        )
    )


#random guessing baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Classification Models"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "classification_roc_curves.png"
)

plt.show()

#decision tree visualization
fitted_preprocessor = (
    decision_tree_pipeline
    .named_steps[
        "preprocessor"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


decision_tree_model = (
    decision_tree_pipeline
    .named_steps[
        "classifier"
    ]
)


plt.figure(
    figsize=(24, 12)
)


plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)


plt.title(
    "Decision Tree - Titanic Survival Prediction"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

#class imbalance
print("\n" + "=" * 60)
print("CLASS IMBALANCE REPORT")
print("=" * 60)


imbalance_report = pd.DataFrame({
    "count": y_train.value_counts().sort_index(),
    "percentage": (
        y_train.value_counts(
            normalize=True
        ).sort_index() * 100
    ).round(2)
})


imbalance_report.index = [
    "Not Survived",
    "Survived"
]


print(imbalance_report)


imbalance_report.to_csv(
    OUTPUT_DIR / "class_balance.csv"
)

#imbalance handling baseline
baseline_imbalance_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(preprocessor)
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


baseline_imbalance_pipeline.fit(
    X_train,
    y_train
)


baseline_predictions = (
    baseline_imbalance_pipeline.predict(
        X_test
    )
)


baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

#imbalance handling class weight
balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(preprocessor)
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


balanced_pipeline.fit(
    X_train,
    y_train
)


balanced_predictions = (
    balanced_pipeline.predict(
        X_test
    )
)


balanced_precision = precision_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

#imbalance handling smote
smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            clone(preprocessor)
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


smote_pipeline.fit(
    X_train,
    y_train
)


smote_predictions = (
    smote_pipeline.predict(
        X_test
    )
)


smote_precision = precision_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_predictions,
    zero_division=0
)

#imbalance strategy comparison
imbalance_comparison = pd.DataFrame({
    "Strategy": [
        "Baseline",
        "Class Weight Balanced",
        "SMOTE"
    ],

    "Precision": [
        baseline_precision,
        balanced_precision,
        smote_precision
    ],

    "Recall": [
        baseline_recall,
        balanced_recall,
        smote_recall
    ],

    "F1": [
        baseline_f1,
        balanced_f1,
        smote_f1
    ]
})


imbalance_comparison[
    [
        "Precision",
        "Recall",
        "F1"
    ]
] = imbalance_comparison[
    [
        "Precision",
        "Recall",
        "F1"
    ]
].round(4)


print("\n" + "=" * 60)
print("IMBALANCE STRATEGY COMPARISON")
print("=" * 60)

print(
    imbalance_comparison.to_string(
        index=False
    )
)


imbalance_comparison.to_csv(
    OUTPUT_DIR
    / "imbalance_comparison.csv",
    index=False
)


best_imbalance_row = (
    imbalance_comparison.loc[
        imbalance_comparison["F1"].idxmax()
    ]
)


print(
    "\nBest imbalance strategy by F1:"
)

print(
    best_imbalance_row["Strategy"]
)

#random forest hyperparameter tuning
tuning_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            clone(preprocessor)
        ),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


parameter_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300
    ],

    "classifier__max_depth": [
        None,
        4,
        6,
        8
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)


print("\nRunning Random Forest GridSearchCV...")


grid_search.fit(
    X_train,
    y_train
)


print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)


print(
    f"\nBest Cross-Validation F1: "
    f"{grid_search.best_score_:.4f}"
)

#oob score
best_tuned_pipeline = (
    grid_search.best_estimator_
)


best_random_forest = (
    best_tuned_pipeline.named_steps[
        "classifier"
    ]
)


oob_score = (
    best_random_forest.oob_score_
)


print(
    f"\nRandom Forest OOB Score: "
    f"{oob_score:.4f}"
)

tuning_results = pd.DataFrame(
    grid_search.cv_results_
)


tuning_results.to_csv(
    OUTPUT_DIR
    / "random_forest_grid_search.csv",
    index=False
)

tuned_forest_results = evaluate_model(
    "Tuned Random Forest",
    best_tuned_pipeline,
    X_test,
    y_test
)

#regression data
regression_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]


regression_categorical_features = [
    "sex",
    "embarked"
]


regression_features = (
    regression_numeric_features
    + regression_categorical_features
)


X_regression = df[
    regression_features
].copy()


y_regression = df[
    "fare"
].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)

#regression preprocessing
regression_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


regression_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            regression_numeric_transformer,
            regression_numeric_features
        ),
        (
            "categorical",
            regression_categorical_transformer,
            regression_categorical_features
        )
    ]
)

#multivariate linear regression
regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


fare_predictions = (
    regression_pipeline.predict(
        X_reg_test
    )
)

#regression metrics
mae = mean_absolute_error(
    y_reg_test,
    fare_predictions
)


mse = mean_squared_error(
    y_reg_test,
    fare_predictions
)


rmse = np.sqrt(
    mse
)


r2 = r2_score(
    y_reg_test,
    fare_predictions
)

transformed_feature_names = (
    regression_pipeline
    .named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
)


n = len(
    y_reg_test
)

p = len(
    transformed_feature_names
)


adjusted_r2 = (
    1
    -
    (
        (1 - r2)
        * (n - 1)
        /
        (n - p - 1)
    )
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)


print(
    f"MAE:         {mae:.4f}"
)

print(
    f"RMSE:        {rmse:.4f}"
)

print(
    f"R²:          {r2:.4f}"
)

print(
    f"Adjusted R²: {adjusted_r2:.4f}"
)

#residuals analysis
residuals = (
    y_reg_test
    - fare_predictions
)


plt.figure(
    figsize=(8, 5)
)


plt.scatter(
    fare_predictions,
    residuals,
    alpha=0.6
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Fare Prediction"
)

plt.tight_layout()


plt.savefig(
    CHART_DIR
    / "fare_regression_residual_plot.png"
)


plt.show()


regression_metrics = pd.DataFrame({
    "Model": [
        "Linear Regression"
    ],

    "MAE": [
        round(
            mae,
            4
        )
    ],

    "RMSE": [
        round(
            rmse,
            4
        )
    ],

    "R2": [
        round(
            r2,
            4
        )
    ],

    "Adjusted_R2": [
        round(
            adjusted_r2,
            4
        )
    ]
})


print(
    "\nRegression Metrics:"
)

print(
    regression_metrics.to_string(
        index=False
    )
)


regression_metrics.to_csv(
    OUTPUT_DIR
    / "regression_metrics.csv",
    index=False
)

#final model comparison table
classification_final = comparison_table.copy()


classification_final[
    "Model Type"
] = "Classification"


classification_final[
    "MAE"
] = np.nan

classification_final[
    "RMSE"
] = np.nan

classification_final[
    "R2"
] = np.nan

classification_final[
    "Adjusted R2"
] = np.nan


regression_final = pd.DataFrame({
    "Model": [
        "Linear Regression"
    ],

    "Accuracy": [
        np.nan
    ],

    "Precision": [
        np.nan
    ],

    "Recall": [
        np.nan
    ],

    "F1": [
        np.nan
    ],

    "AUC": [
        np.nan
    ],

    "Model Type": [
        "Regression"
    ],

    "MAE": [
        round(
            mae,
            4
        )
    ],

    "RMSE": [
        round(
            rmse,
            4
        )
    ],

    "R2": [
        round(
            r2,
            4
        )
    ],

    "Adjusted R2": [
        round(
            adjusted_r2,
            4
        )
    ]
})


final_model_comparison = pd.concat(
    [
        classification_final,
        regression_final
    ],
    ignore_index=True
)


final_model_comparison = final_model_comparison[
    [
        "Model Type",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "MAE",
        "RMSE",
        "R2",
        "Adjusted R2"
    ]
]


print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)


print(
    final_model_comparison.to_string(
        index=False
    )
)


final_model_comparison.to_csv(
    OUTPUT_DIR
    / "final_model_comparison.csv",
    index=False
)

#best classification pipeline
pipeline_candidates = {
    "Logistic Regression": {
        "pipeline":
            logistic_pipeline,

        "f1":
            logistic_results[
                "f1"
            ]
    },

    "Decision Tree": {
        "pipeline":
            decision_tree_pipeline,

        "f1":
            tree_results[
                "f1"
            ]
    },

    "Random Forest": {
        "pipeline":
            random_forest_pipeline,

        "f1":
            forest_results[
                "f1"
            ]
    },

    "Tuned Random Forest": {
        "pipeline":
            best_tuned_pipeline,

        "f1":
            tuned_forest_results[
                "f1"
            ]
    }
}


best_model_name = max(
    pipeline_candidates,
    key=lambda model_name:
        pipeline_candidates[
            model_name
        ]["f1"]
)


best_pipeline = (
    pipeline_candidates[
        best_model_name
    ]["pipeline"]
)


best_pipeline_f1 = (
    pipeline_candidates[
        best_model_name
    ]["f1"]
)


print(
    "\nBest model based on F1:"
)

print(
    best_model_name
)

print(
    f"F1 Score: "
    f"{best_pipeline_f1:.4f}"
)

#save complete pipeline
MODEL_FILE = (
    BASE_DIR
    / "best_model_pipeline.joblib"
)


joblib.dump(
    best_pipeline,
    MODEL_FILE
)


print(
    "\nComplete pipeline saved to:"
)

print(
    MODEL_FILE
)

loaded_pipeline = joblib.load(
    MODEL_FILE
)


print(
    "\nSaved pipeline loaded successfully."
)


raw_test_sample = (
    X_test.head(5).copy()
)


reloaded_predictions = (
    loaded_pipeline.predict(
        raw_test_sample
    )
)


print(
    "\nRaw Test Input:"
)

print(
    raw_test_sample
)


print(
    "\nPredictions from Reloaded Pipeline:"
)

print(
    reloaded_predictions
)

original_predictions = (
    best_pipeline.predict(
        raw_test_sample
    )
)


predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions
)


print(
    "\nOriginal and Reloaded "
    "Predictions Match:"
)

print(
    predictions_match
)


print(
    f"\nBest Saved Classifier: "
    f"{best_model_name}"
)

print(
    f"Best F1 Score: "
    f"{best_pipeline_f1:.4f}"
)

print(
    f"Random Forest OOB Score: "
    f"{oob_score:.4f}"
)

print(
    f"Regression MAE: "
    f"{mae:.4f}"
)

print(
    f"Regression RMSE: "
    f"{rmse:.4f}"
)

print(
    f"Regression R²: "
    f"{r2:.4f}"
)

print(
    f"Regression Adjusted R²: "
    f"{adjusted_r2:.4f}"
)