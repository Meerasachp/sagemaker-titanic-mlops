# train.py
import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # SageMaker input directories
    input_path = os.path.join("/opt/ml/input/data/train", "train.csv")
    model_path = os.path.join("/opt/ml/model")

    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure target column exists
    if "Survived" not in df.columns:
        raise ValueError(f"'Survived' column not found. Available: {df.columns}")

    # Drop irrelevant columns
    drop_cols = ["Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle categorical features (Sex, Embarked)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Features and labels
    y = df["Survived"]
    X = df.drop("Survived", axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost classifier
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "xgboost-model.json")
    clf.save_model(model_file)
    print(f"Model saved at: {model_file}")

