import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_data(df: pd.DataFrame) -> bool:

    # 1. Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Check duplicate customer IDs
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID found.")

    # 3. Check numeric ranges
    if (df["SeniorCitizen"] < 0).any():
        raise ValueError("SeniorCitizen cannot be negative.")

    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative.")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges cannot be negative.")

    # 4. Check Churn values
    allowed_churn_values = {"Yes", "No"}

    if not set(df["Churn"].unique()).issubset(allowed_churn_values):
        raise ValueError("Invalid Churn value found.")

    # 5. Check blank TotalCharges
    blank_total_charges = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Blank TotalCharges values: {blank_total_charges}")

    print("Data validation passed!")

    return True