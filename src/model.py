import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score 


FEATURE_COLS = ["visit_count", "tenure_days", "avg_gap_days",
                "total_spend", "order_count", "age", "sex", "ReaMonths"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode and order features exactly as the model expects.

    The single home for training/serving-consistent preparation:
    every caller (training, API, batch scoring) goes through this,
    so encoding can never drift between them.
    """
    X = df.copy()
    X["sex"] = (X["sex"] == "M").astype(int)
    return X[FEATURE_COLS]




def build_model(features) -> tuple:
    """Temporal split on index_visit, train XGBoost, return (model, metrics)."""


    model_df = features[FEATURE_COLS + ["churned", "index_visit"]].copy()
    model_df = model_df.sort_values("index_visit")
    cutoff = model_df["index_visit"].quantile(0.8)
    train = model_df[model_df["index_visit"] < cutoff]
    test  = model_df[model_df["index_visit"] >= cutoff]

    X_train, y_train = prepare_features(train), train["churned"]
    X_test,  y_test  = prepare_features(test),  test["churned"]

    model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=2.0,
            eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    test_scores = pd.DataFrame({
    "churn_prob": model.predict_proba(X_test)[:, 1],
    "churned": y_test,
    "index_visit": test["index_visit"],
    }, index=test.index)

    metrics = {
        "train_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        "test_auc":  roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1]),
    }
    return model, metrics, test_scores