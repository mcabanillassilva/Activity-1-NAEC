import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def treat_outliers(df, num_cols):
    df_out = df.copy()
    treated_summary = {}

    for col in num_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply outlier
        outliers_lower = df_out[df_out[col] < lower_bound].shape[0]
        outliers_upper = df_out[df_out[col] > upper_bound].shape[0]
        treated_summary[col] = outliers_lower + outliers_upper

        # Replace values with outliers
        df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)

    return df_out, treated_summary

def main():
    df = pd.read_csv("AmesHousing.csv")
    print("Initial dimension:", df.shape)

    # Target variable
    target = "SalePrice"

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Numeric:", len(num_cols))
    #print("Numeric Variables: ", num_cols)
    print("Categorical:", len(cat_cols))
    #print("Categorical Variables: ", cat_cols)

    # Add missing values for numerical columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Add missing values for categorical columns
    for col in cat_cols:
        df[col] = df[col].replace("NA", "Missing")
        df[col] = df[col].fillna("Missing")
        
    # Treat outliers
    df, outliers_treated = treat_outliers(df, num_cols)
    print("Number of outliers treated per numerical column:")
    for col, count in outliers_treated.items():
        print(f"    {col}: {count}")

    # One hot encoding for categorical variables
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("Dimension after One Hot Encoding", df.shape)

    # Log-transform the target variable
    df[target] = np.log1p(df[target])

    # Inputs
    X = df.drop(columns=[target])
    # Output
    y = df[target]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=42
    )

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, shuffle=True, random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    num_cols_in_X = [col for col in X_train.columns if col in num_cols]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled   = X_val.copy()
    X_test_scaled  = X_test.copy()

    # Scale just numerical features
    X_train_scaled[num_cols_in_X] = scaler.fit_transform(X_train[num_cols_in_X])
    X_val_scaled[num_cols_in_X]   = scaler.transform(X_val[num_cols_in_X])
    X_test_scaled[num_cols_in_X]  = scaler.transform(X_test[num_cols_in_X])

    # Save processed datasets
    os.makedirs("output", exist_ok=True)

    train_final = X_train_scaled.copy()
    train_final[target] = y_train.values
    train_final.to_csv("output/train_processed.csv", index=False)

    val_final = X_val_scaled.copy()
    val_final[target] = y_val.values
    val_final.to_csv("output/val_processed.csv", index=False)

    test_final = X_test_scaled.copy()
    test_final[target] = y_test.values
    test_final.to_csv("output/test_processed.csv", index=False)

    # Save scaler
    with open("part_1/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved files:")
    print("- train_processed.csv")
    print("- val_processed.csv")
    print("- test_processed.csv")
    print("- scaler.pkl")

if __name__ == "__main__":
    main()