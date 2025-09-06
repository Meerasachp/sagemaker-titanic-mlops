import argparse
import os
import pandas as pd
import xgboost as xgb

def model_fn(model_dir):
    """Load model for inference"""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model

if __name__ == "__main__":
    # Parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    args = parser.parse_args()

    # Load training data
    train_data = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_data = pd.read_csv(os.path.join(args.validation, "test.csv"))

    # Split features/labels
    y_train = train_data.pop("Survived")
    X_train = train_data
    y_val = val_data.pop("Survived")
    X_val = val_data

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train model
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "eval_metric": "auc"
    }
    evals = [(dtrain, "train"), (dval, "validation")]

    model = xgb.train(params, dtrain, num_boost_round=args.num_round, evals=evals)

    # Save model
    model.save_model(os.path.join(args.model_dir, "xgboost-model"))

