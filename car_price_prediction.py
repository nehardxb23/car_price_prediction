import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# 1. Load CarDekho dataset
# -----------------------------------

df = pd.read_csv("cardekho_dataset.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(df))


# -----------------------------------
# 2. Select features
# -----------------------------------

features = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

target = "selling_price"

df = df[features + [target]]

# Remove rows with missing values
df = df.dropna()

print("Rows after cleaning:", len(df))


# -----------------------------------
# 3. Separate input and target
# -----------------------------------

X = df[features]
y = df[target]


# -----------------------------------
# 4. Define categorical and numerical features
# -----------------------------------

categorical_features = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

numerical_features = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# -----------------------------------
# 5. Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# -----------------------------------
# 6. Create ML model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# 7. Create complete pipeline
# -----------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------
# 8. Split data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# 9. Start MLflow experiment
# -----------------------------------

mlflow.set_experiment("Car_Price_Prediction")


with mlflow.start_run():

    # -----------------------------------
    # 10. Train model
    # -----------------------------------

    pipeline.fit(X_train, y_train)

    print("Model training completed!")


    # -----------------------------------
    # 11. Make predictions
    # -----------------------------------

    predictions = pipeline.predict(X_test)


    # -----------------------------------
    # 12. Calculate evaluation metrics
    # -----------------------------------

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )


    # -----------------------------------
    # 13. Model validation
    # -----------------------------------

    if r2 >= 0.80:
        validation_status = "PASSED"
    else:
        validation_status = "FAILED"


    # -----------------------------------
    # 14. Log parameters to MLflow
    # -----------------------------------

    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "features",
        len(features)
    )


    # -----------------------------------
    # 15. Log metrics to MLflow
    # -----------------------------------

    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "RMSE",
        rmse
    )

    mlflow.log_metric(
        "R2",
        r2
    )


    # -----------------------------------
    # 16. Log validation status
    # -----------------------------------

    mlflow.set_tag(
        "validation_status",
        validation_status
    )


    # -----------------------------------
    # 17. Log trained model
    # -----------------------------------

    mlflow.sklearn.log_model(
        pipeline,
        "car_price_model"
    )


    # -----------------------------------
    # 18. Display results
    # -----------------------------------

    print("\n-----------------------------")
    print("MODEL EVALUATION")
    print("-----------------------------")

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Validation:", validation_status)

    print("-----------------------------")