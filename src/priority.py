
import pandas as pd

def build_priority(churn_probs, values):
    df = pd.DataFrame({"churn_prob": churn_probs, "value": values})
    df["priority"] = df["churn_prob"] * df["value"]
    df = df.sort_values("priority", ascending=False)
    df["cumulative_expected"] = df["priority"].cumsum()
    return df