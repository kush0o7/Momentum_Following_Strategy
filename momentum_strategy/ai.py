from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    down = -delta.clip(upper=0).rolling(window=period, min_periods=period).mean()
    rs = up / down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


@dataclass(frozen=True)
class AIResult:
    model_name: str
    accuracy: float
    latest_score: float
    message: str


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["ret_1d"] = df["close"].pct_change()
    feats["ret_5d"] = df["close"].pct_change(5)
    feats["sma_20"] = df["close"].rolling(20).mean()
    feats["sma_50"] = df["close"].rolling(50).mean()
    feats["sma_slope"] = feats["sma_20"].pct_change()
    feats["rsi"] = _rsi(df["close"], 14)
    feats["atr_pct"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"]
    feats = feats.dropna()
    return feats


def train_ai_model(df: pd.DataFrame) -> AIResult:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except Exception:
        return AIResult(
            model_name="None",
            accuracy=float("nan"),
            latest_score=float("nan"),
            message="scikit-learn not installed; install it to enable AI scoring.",
        )

    feats = build_features(df)
    if feats.empty or len(feats) < 200:
        return AIResult(
            model_name="LogisticRegression",
            accuracy=float("nan"),
            latest_score=float("nan"),
            message="Not enough data to train AI model.",
        )

    future_ret = df["close"].pct_change().shift(-1).reindex(feats.index)
    y = (future_ret > 0).astype(int)

    split = int(len(feats) * 0.8)
    X_train, X_test = feats.iloc[:split], feats.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    latest_score = float(model.predict_proba(feats.iloc[[-1]])[0][1])

    return AIResult(
        model_name="LogisticRegression",
        accuracy=float(acc),
        latest_score=latest_score,
        message="AI score is probability of next-day positive return.",
    )
