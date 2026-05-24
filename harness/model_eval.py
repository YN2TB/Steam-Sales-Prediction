"""
harness/model_eval.py
Time-split evaluation for baseline and two-stage models.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Thêm thư mục parent vào sys.path để import từ các module ngoài
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

from config.settings import MODELS_DIR
from harness.tool_harness import tool_harness
from tools.bucket_model import assign_bucket, train_bucket_models
from tools.data_tool import load_sales_data, clean_sales_data
from tools.decoupled_model import (
    FEATURE_COLUMNS,
    build_time_series_features,
    predict_sale_event_batch,
    train_two_stage_models,
)
from tools.feature_tool import build_features
from tools.model_tool import predict_on_features, train



@dataclass
class MetricSummary:
    window_f1: float
    window_pr_auc: float
    discount_mae: float
    duration_mae: float


def _split_by_date(
    steam_df: pd.DataFrame, split_ratio: float
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(steam_df["date"].sort_values().unique())
    if dates.empty:
        raise ValueError("No dates available for split")

    cutoff_idx = max(0, int(len(dates) * split_ratio) - 1)
    cutoff_date = pd.to_datetime(dates.iloc[cutoff_idx])

    return cutoff_date, steam_df[steam_df["date"] <= cutoff_date].copy(), steam_df[
        steam_df["date"] > cutoff_date
    ].copy()


def _compute_window_metrics(y_true_window: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    pred_window = scores >= 0.5
    f1 = f1_score(y_true_window, pred_window)
    pr_auc = average_precision_score(y_true_window, scores)
    return float(f1), float(pr_auc)


def _baseline_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    app_id: str,
) -> MetricSummary:
    train_features = build_features(train_df)["data"]
    test_features = build_features(test_df)["data"]

    if len(train_features) < 5 or test_features.empty:
        raise ValueError("Not enough samples for baseline evaluation")

    train(train_features, app_id)
    preds = predict_on_features(test_features, app_id)["data"]

    pred_gap = np.array([p["predicted_gap_days"] for p in preds], dtype=float)
    pred_discount = np.array([p["predicted_discount"] for p in preds], dtype=float)

    y_true_gap = test_features["next_gap_days"].to_numpy(dtype=float)
    y_true_discount = test_features["next_discount"].to_numpy(dtype=float)

    y_true_window = (y_true_gap <= 7).astype(int)
    scores = np.clip(1.0 - (pred_gap / 7.0), 0.0, 1.0)
    f1, pr_auc = _compute_window_metrics(y_true_window, scores)

    discount_mae = mean_absolute_error(y_true_discount, pred_discount)
    return MetricSummary(
        window_f1=f1,
        window_pr_auc=pr_auc,
        discount_mae=float(discount_mae),
        duration_mae=0.0,
    )


def _two_stage_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    app_id: str,
) -> MetricSummary:
    train_two_stage_models(train_df, app_id)

    test_features = build_time_series_features(test_df)["data"]

    if test_features.empty:
        raise ValueError("Not enough samples for two-stage evaluation")

    batch_preds = predict_sale_event_batch(test_features, app_id)["data"]

    pred_prob = np.array([p["probability"] for p in batch_preds], dtype=float)
    pred_discount = np.array(
        [p["predicted_discount_pct"] for p in batch_preds], dtype=float
    )
    pred_duration = np.array(
        [p["predicted_duration_days"] for p in batch_preds], dtype=float
    )

    y_true_window = (test_features["days_to_next_sale"].fillna(0) > 0).to_numpy(dtype=int)
    y_true_discount = test_features["target_discount_pct"].to_numpy(dtype=float)
    y_true_duration = test_features["target_duration_days"].to_numpy(dtype=float)

    f1, pr_auc = _compute_window_metrics(y_true_window, pred_prob)
    discount_mae = mean_absolute_error(y_true_discount, pred_discount)
    duration_mae = mean_absolute_error(y_true_duration, pred_duration)

    return MetricSummary(
        window_f1=f1,
        window_pr_auc=pr_auc,
        discount_mae=float(discount_mae),
        duration_mae=float(duration_mae),
    )

def _mean_metrics(scores: Iterable[MetricSummary]) -> dict:
    if not scores:
        return {
            "window_f1": 0.0,
            "window_pr_auc": 0.0,
            "discount_mae": 0.0,
            "duration_mae": 0.0,
        }
    return {
        "window_f1": round(float(np.mean([s.window_f1 for s in scores])), 3),
        "window_pr_auc": round(float(np.mean([s.window_pr_auc for s in scores])), 3),
        "discount_mae": round(float(np.mean([s.discount_mae for s in scores])), 3),
        "duration_mae": round(float(np.mean([s.duration_mae for s in scores])), 3),
    }


def _bucket_model_path(bucket: str) -> Path:
    return MODELS_DIR / "buckets" / f"{bucket}_sale_window_xgb.pkl"


def _bucketed_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    app_id: str,
) -> MetricSummary:
    bucket = assign_bucket(train_df)
    if bucket is None:
        raise ValueError("Game history < 1 year; bucketed eval skipped")

    path = _bucket_model_path(bucket)
    if not path.exists():
        raise FileNotFoundError(f"Bucket model not found: {bucket}")

    with open(path, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]

    test_features = build_time_series_features(test_df)["data"]
    if test_features.empty:
        raise ValueError("Not enough samples for bucketed evaluation")

    X_all = test_features[FEATURE_COLUMNS].values
    prob_all = clf.predict_proba(X_all)
    prob = 1.0 - prob_all[:, 0]

    y_true_window = (test_features["days_to_next_sale"].fillna(0) > 0).to_numpy(dtype=int)
    y_true_discount = test_features["target_discount_pct"].to_numpy(dtype=float)
    y_true_duration = test_features["target_duration_days"].to_numpy(dtype=float)

    f1, pr_auc = _compute_window_metrics(y_true_window, prob)

    mean_discount = float(y_true_discount.mean()) if len(y_true_discount) else 0.0
    pred_discount = np.full_like(y_true_discount, mean_discount, dtype=float)
    discount_mae = mean_absolute_error(y_true_discount, pred_discount)

    mean_duration = float(y_true_duration.mean()) if len(y_true_duration) else 0.0
    pred_duration = np.full_like(y_true_duration, mean_duration, dtype=float)
    duration_mae = mean_absolute_error(y_true_duration, pred_duration)

    return MetricSummary(
        window_f1=f1,
        window_pr_auc=pr_auc,
        discount_mae=float(discount_mae),
        duration_mae=float(duration_mae),
    )


def _evaluate_mode(df: pd.DataFrame, split_ratio: float, mode: str) -> dict:
    baseline_scores = []
    two_stage_scores = []
    bucketed_scores = []

    games = (
        df.groupby(["app_id", "game_title"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
    )

    if mode == "steam_only":
        train_bucket_models(df[df["shop_name"].str.lower() == "steam"])
    elif mode == "all_train_steam_test":
        train_bucket_models(df)

    for _, g in games.iterrows():
        game_df = df[df["app_id"] == g["app_id"]].sort_values("date")
        if len(game_df) < 10:
            continue

        steam_mask = game_df["shop_name"].str.lower() == "steam"
        steam_df = game_df[steam_mask]
        if len(steam_df) < 10:
            continue

        cutoff_date, _, _ = _split_by_date(steam_df, split_ratio)

        if mode == "steam_only":
            train_df = steam_df[steam_df["date"] <= cutoff_date]
        elif mode == "all_train_steam_test":
            train_df = game_df[game_df["date"] <= cutoff_date]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        test_df = steam_df[steam_df["date"] > cutoff_date]

        try:
            b = _baseline_eval(train_df, test_df, g["app_id"])
            baseline_scores.append(b)
        except Exception:
            continue

        try:
            t = _two_stage_eval(train_df, test_df, g["app_id"])
            two_stage_scores.append(t)
        except Exception:
            continue

        try:
            bkt = _bucketed_eval(train_df, test_df, g["app_id"])
            bucketed_scores.append(bkt)
        except Exception:
            continue

    return {
        "baseline": _mean_metrics(baseline_scores),
        "two_stage": _mean_metrics(two_stage_scores),
        "bucketed": _mean_metrics(bucketed_scores),
        "games_evaluated": int(max(len(baseline_scores), len(two_stage_scores))),
    }


@tool_harness("model_eval")
def evaluate_models(df: pd.DataFrame, split_ratio: float = 0.8) -> dict:
    """
    Evaluate baseline and two-stage models using time-based splits.
    Returns both Steam-only and all-train/Steam-test modes.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return {
        "steam_only": _evaluate_mode(df, split_ratio, mode="steam_only"),
        "all_train_steam_test": _evaluate_mode(
            df, split_ratio, mode="all_train_steam_test"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Time split evaluation")
    p.add_argument("--csv", default="TS_price_history.csv")
    p.add_argument("--split", type=float, default=0.8)
    args = p.parse_args()

    df = clean_sales_data(load_sales_data(args.csv)["data"])["data"]
    print(df.columns.tolist())
    print("\nSteam-only data time split evaluation")
    print("-" * 40)
    result = evaluate_models(df, split_ratio=args.split)["data"]

    def _print_block(label: str, data: dict) -> None:
        print(f"\n{label}:")
        print("  Baseline:")
        print(f"    Window F1: {data['baseline']['window_f1']}")
        print(f"    Window PR-AUC: {data['baseline']['window_pr_auc']}")
        print(f"    Discount MAE: {data['baseline']['discount_mae']}")
        print("  Two-stage:")
        print(f"    Window F1: {data['two_stage']['window_f1']}")
        print(f"    Window PR-AUC: {data['two_stage']['window_pr_auc']}")
        print(f"    Discount MAE: {data['two_stage']['discount_mae']}")
        print(f"    Duration MAE: {data['two_stage']['duration_mae']}")
        print("  Bucketed:")
        print(f"    Window F1: {data['bucketed']['window_f1']}")
        print(f"    Window PR-AUC: {data['bucketed']['window_pr_auc']}")
        print(f"    Discount MAE: {data['bucketed']['discount_mae']}")
        print(f"    Duration MAE: {data['bucketed']['duration_mae']}")
        print(f"  Games evaluated: {data['games_evaluated']}")

    _print_block("Steam-only train/test", result["steam_only"])
    _print_block("Train all shops, test Steam", result["all_train_steam_test"])


if __name__ == "__main__":
    main()
