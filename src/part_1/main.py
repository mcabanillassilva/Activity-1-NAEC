import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def treat_outliers(df, num_cols):
    df_out = df.copy()
    treated_summary = {}

    for col in num_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_lower = df_out[df_out[col] < lower_bound].shape[0]
        outliers_upper = df_out[df_out[col] > upper_bound].shape[0]
        treated_summary[col] = outliers_lower + outliers_upper

        df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)

    return df_out, treated_summary

def main():
    df = pd.read_csv("AmesHousing.csv")
    print("Initial dimension:", df.shape)

    target = "SalePrice"
    df_ids = df[["PID"]].copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].replace("NA", "Missing").fillna("Missing")

    # Treat outliers
    df, outliers_treated = treat_outliers(df, num_cols)
    print("Number of outliers treated per numerical column:")
    for col, count in outliers_treated.items():
        print(f"    {col}: {count}")

    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    print("Dimension after One Hot Encoding:", df.shape)

    # Inputs and target
    X = df.drop(columns=[target])
    y = df[target]

    # Split train/validation/test with PID
    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, df_ids, test_size=0.20, shuffle=True, random_state=42
    )
    X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
        X_train, y_train, pid_train, test_size=0.20, shuffle=True, random_state=42
    )

    # Guardar directamente sin escalar
    train_final = X_train.copy()
    train_final[target] = y_train
    train_final['PID'] = pid_train.values

    val_final = X_val.copy()
    val_final[target] = y_val
    val_final['PID'] = pid_val.values

    test_final = X_test.copy()
    test_final[target] = y_test
    test_final['PID'] = pid_test.values

    os.makedirs("output", exist_ok=True)
    train_final.to_csv("output/train_processed.csv", index=False)
    val_final.to_csv("output/val_processed.csv", index=False)
    test_final.to_csv("output/test_processed.csv", index=False)

    print("Saved files:")
    print("- train_processed.csv")
    print("- val_processed.csv")
    print("- test_processed.csv")

if __name__ == "__main__":
    main()
