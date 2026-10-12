"""Hàm tạo đặc trưng dùng chung cho train_stacking.py và app.py."""
import pandas as pd

NUM_FEATURES = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
]
CAT_FEATURES = ["type"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 2 đặc trưng chênh lệch số dư (sai lệch so với giao dịch hợp lệ)."""
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df
