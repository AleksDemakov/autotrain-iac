import yaml, mlflow, pandas as pd
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient

MODEL_NAME = "kaggle-champion"
cfg = yaml.safe_load(open("config/models.yaml"))

df = pd.read_csv(cfg["dataset"])
X, y = df.drop(columns=[cfg["target"]]), df[cfg["target"]]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=cfg["seed"])  # seed!

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def make_pipe(model, X):
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    return Pipeline([("prep", preprocessor), ("model", model)])


def promote_if_better(version, auc):
    client = MlflowClient()
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ_auc = client.get_run(champ.run_id).data.metrics["val_roc_auc"]
        if auc <= champ_auc:
            print(f"champion удержан (new {auc:.4f} <= {champ_auc:.4f})")
            return
    except Exception:
        pass  
    client.set_registered_model_alias(MODEL_NAME, "champion", version)

mlflow.sklearn.autolog()
for m in cfg["models"]:
    cls = getattr(import_module(m["class"].rsplit(".", 1)[0]),
                  m["class"].rsplit(".", 1)[1])
    pipe = make_pipe(cls(**m.get("params", {})), X_tr)
    with mlflow.start_run(run_name=m["name"]):
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        mlflow.log_metric("val_roc_auc", auc) 
        mv = mlflow.sklearn.log_model(pipe, "model",
                                      registered_model_name=MODEL_NAME, serialization_format="cloudpickle")
        promote_if_better(mv.registered_model_version, auc)
