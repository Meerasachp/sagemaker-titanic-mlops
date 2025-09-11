# src/evaluate.py
import argparse, json, os, tarfile, tempfile, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
import boto3

def download_model_tar(s3_uri, dst_dir):
    bucket, key = s3_uri.replace("s3://","").split("/", 1)
    local_tar = os.path.join(dst_dir, "model.tar.gz")
    boto3.client("s3").download_file(bucket, key, local_tar)
    with tarfile.open(local_tar, "r:gz") as t:
        t.extractall(dst_dir)
    return os.path.join(dst_dir, "xgboost-model")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)               # S3 or local path to test.csv (label first col, no header)
    p.add_argument("--model_artifact", required=True)     # S3 model.tar.gz from training
    p.add_argument("--out", default="/opt/ml/processing/output/metrics.json")
    args = p.parse_args()

    # Load test
    df = pd.read_csv(args.test, header=None)
    y, X = df.iloc[:,0], df.iloc[:,1:]

    # Load model
    tmp = tempfile.mkdtemp()
    model_path = download_model_tar(args.model_artifact, tmp)
    booster = xgb.Booster(); booster.load_model(model_path)

    # Predict
    dtest = xgb.DMatrix(X)
    proba = booster.predict(dtest)

    metrics = {
        "auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, (proba>=0.5).astype(int)))
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"))
    print(json.dumps(metrics))

