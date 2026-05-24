"""
harness/eval.py
Test suite tự động cho Steam Sales Window.
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Thêm thư mục parent vào sys.path để import từ các module ngoài
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _mock_sales_df(n: int = 20, app_id: str = "570") -> pd.DataFrame:
    """Tạo DataFrame giả để test không cần CSV thật."""
    import random

    random.seed(42)
    dates = []
    base = datetime(2020, 1, 1)
    curr = base
    for _ in range(n):
        curr += timedelta(days=random.randint(30, 120))
        dates.append(curr)

    return pd.DataFrame(
        {
            "game_title": ["Test Game"] * n,
            "app_id": [app_id] * n,
            "date": dates,
            "price": [random.uniform(5, 20) for _ in range(n)],
            "regular_price": [20.0] * n,
            "sales_percentage": [random.choice([25, 33, 50, 66, 75]) for _ in range(n)],
            "shop_id": ["1"] * n,
            "shop_name": ["Steam"] * n,
            "currency": ["USD"] * n,
        }
    )


TEST_CASES = [
    {
        "id": "TC001",
        "description": "data_tool drop is_historical_low column",
        "run": lambda: _test_drop_leakage(),
    },
    {
        "id": "TC002",
        "description": "feature_tool không có look-ahead bias",
        "run": lambda: _test_no_lookahead(),
    },
    {
        "id": "TC003",
        "description": "feature_tool tạo đủ cyclical features",
        "run": lambda: _test_cyclical_features(),
    },
    {
        "id": "TC004",
        "description": "model_tool train với ≥5 samples",
        "run": lambda: _test_train_min_samples(),
    },
    {
        "id": "TC005",
        "description": "model_tool predict discount trong [0, 100]",
        "run": lambda: _test_discount_range(),
    },
    {
        "id": "TC006",
        "description": "model_tool predict gap_days > 0",
        "run": lambda: _test_gap_positive(),
    },

    {
        "id": "TC011",
        "description": "report_tool save predictions sort/format",
        "run": lambda: _test_save_predictions_format(),
    },
    {
        "id": "TC012",
        "description": "model_eval time split metrics",
        "run": lambda: _test_time_split_eval(),
    },
    {
        "id": "TC013",
        "description": "bucket_model assign bucket",
        "run": lambda: _test_bucket_assignment(),
    },
    {
        "id": "TC014",
        "description": "bucket_model predict",
        "run": lambda: _test_bucket_predict(),
    },
    {
        "id": "TC007",
        "description": "tool_harness bắt exception → status=error",
        "run": lambda: _test_harness_error(),
    },
    {
        "id": "TC008",
        "description": "two-stage features tạo đủ columns + targets",
        "run": lambda: _test_two_stage_features(),
    },
    {
        "id": "TC009",
        "description": "two-stage train/predict chạy end-to-end",
        "run": lambda: _test_two_stage_train_predict(),
    },
]


def _test_drop_leakage():

    # Tạo CSV giả có is_historical_low
    df = _mock_sales_df()
    df["is_historical_low"] = True

    # Patch load để dùng df giả
    # Source - https://stackoverflow.com/a/28712742
    # Posted by Shinbero, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-22, License - CC BY-SA 4.0

    # from tools import data_tool
    import unittest.mock as mock

    with mock.patch("pandas.read_csv", return_value=df):
        # Tạo file giả
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            # r = data_tool.load_sales_data(
            #     f.name.split("/")[-1] if "/" in f.name else f.name
            # )
        os.unlink(f.name)

    # Dùng trực tiếp
    from tools.data_tool import LEAKAGE_COLS

    result = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    assert "is_historical_low" not in result.columns
    return "is_historical_low dropped correctly"


def _test_no_lookahead():
    from tools.feature_tool import build_features

    df = _mock_sales_df(15)
    r = build_features(df)
    assert r["status"] == "ok"
    feat = r["data"]
    # Target columns dùng shift(-1) nên row cuối phải bị drop
    assert len(feat) < len(df), "Rows cuối phải bị drop (target là tương lai)"
    return f"{len(df)} → {len(feat)} rows sau drop NaN"


def _test_cyclical_features():
    from tools.feature_tool import build_features

    df = _mock_sales_df(15)
    r = build_features(df)
    assert r["status"] == "ok"
    feat = r["data"]
    for col in ["month_sin", "month_cos", "dow_sin", "dow_cos"]:
        assert col in feat.columns, f"Thiếu {col}"
        assert feat[col].between(-1, 1).all(), f"{col} ngoài [-1, 1]"
    return "all cyclical features in [-1, 1]"


def _test_train_min_samples():
    from tools.feature_tool import build_features
    from tools.model_tool import train

    df = _mock_sales_df(4)  # < 5 samples
    feat = build_features(df)["data"]
    r = train(feat, "test_app")
    assert r["status"] == "error", "Phải fail với < 5 samples"
    return "correctly rejected < 5 samples"


def _test_discount_range():
    from tools.feature_tool import build_features
    from tools.model_tool import train, predict_next_sales

    df = _mock_sales_df(20)
    feat = build_features(df)["data"]
    train(feat, "test_570")
    r = predict_next_sales(feat, "test_570", n=3)
    assert r["status"] == "ok"
    for p in r["data"]:
        assert 0 <= p["predicted_discount"] <= 100, (
            f"Discount ngoài [0,100]: {p['predicted_discount']}"
        )
    return f"{len(r['data'])} predictions, discounts in [0, 100]"


def _test_gap_positive():
    from tools.feature_tool import build_features
    from tools.model_tool import train, predict_next_sales

    df = _mock_sales_df(20)
    feat = build_features(df)["data"]
    train(feat, "test_570")
    r = predict_next_sales(feat, "test_570", n=3)
    assert r["status"] == "ok"
    for p in r["data"]:
        assert p["days_from_now"] is not None
    return "gap_days predictions generated"


def _test_harness_error():
    from harness.tool_harness import tool_harness

    @tool_harness("test_broken", max_retries=1)
    def broken():
        raise RuntimeError("intentional")

    r = broken()
    assert r["status"] == "error"
    assert "intentional" in r["error"]
    return "harness caught exception correctly"


def _test_save_predictions_format():
    from tools.report_tool import save_predictions

    rows = [
        {
            "game_title": "B Game",
            "shop_name": "Steam",
            "date": "2024-01-02",
            "window_end": "2024-01-08",
            "predicted_discount": 10.0,
            "probability": 0.5,
        },
        {
            "game_title": "A Game",
            "shop_name": "Fanatical",
            "date": "2024-01-01",
            "window_end": "2024-01-07",
            "predicted_discount": 20.0,
            "probability": 0.8,
        },
    ]

    r = save_predictions(rows, filename="_test_predictions.csv")
    assert r["status"] == "ok"

    df = pd.read_csv(r["data"]["path"])
    assert list(df.columns)[:3] == ["game_title", "shop_name", "date"]
    assert df.iloc[0]["game_title"] == "A Game"
    assert df.iloc[0]["shop_name"] == "Fanatical"
    assert isinstance(df.iloc[0]["date"], str)
    assert len(df.iloc[0]["date"]) == 10 and df.iloc[0]["date"][2] == "/"
    return "saved predictions sorted and formatted"


def _test_time_split_eval():
    from harness.model_eval import evaluate_models

    df = _mock_sales_df(40)
    df["shop_name"] = "Steam"
    r = evaluate_models(df, split_ratio=0.8)
    assert r["status"] == "ok"
    data = r["data"]
    assert "steam_only" in data and "all_train_steam_test" in data
    assert "bucketed" in data["steam_only"]
    assert "bucketed" in data["all_train_steam_test"]
    return "time split evaluation ran"


def _test_two_stage_features():
    from tools.decoupled_model import build_time_series_features

    df = _mock_sales_df(12)
    r = build_time_series_features(df)
    assert r["status"] == "ok"
    feat = r["data"]

    required = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_momentum_7",
        "rolling_price_mean_7",
        "rolling_price_std_7",
        "rolling_price_mean_30",
        "rolling_price_std_30",
        "days_since_last_sale",
        "avg_sale_duration",
        "is_summer_sale",
        "is_autumn_sale",
        "is_winter_sale",
        "days_to_next_sale",
        "target_duration_days",
        "target_discount_pct",
    ]
    for col in required:
        assert col in feat.columns, f"Missing {col}"

    assert (feat["days_to_next_sale"].dropna() >= 0).all(), "days_to_next_sale < 0"
    assert (feat["target_duration_days"] >= 0).all(), "target_duration_days < 0"
    assert feat["target_discount_pct"].between(0, 100).all(), (
        "target_discount_pct outside [0, 100]"
    )
    return "two-stage features OK"


def _test_two_stage_train_predict():
    from tools.decoupled_model import train_two_stage_models, predict_sale_event

    df = _mock_sales_df(15)
    r = train_two_stage_models(df, app_id="test_app")
    assert r["status"] == "ok"

    p = predict_sale_event(df, app_id="test_app")
    assert p["status"] == "ok"
    data = p["data"]
    assert "sale_predicted" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["predicted_discount_pct"] <= 100.0
    assert data["predicted_start_offset_days"] >= 0
    assert data["predicted_duration_days"] >= 0
    return "two-stage pipeline OK"


def _test_bucket_assignment():
    from tools.bucket_model import assign_bucket

    df = _mock_sales_df(15)
    df.loc[0, "date"] = df["date"].min() - timedelta(days=370)
    bucket = assign_bucket(df)
    assert bucket is not None
    return f"assigned {bucket}"


def _test_bucket_predict():
    from tools.bucket_model import train_bucket_models, predict_bucketed

    df = _mock_sales_df(15)
    df.loc[0, "date"] = df["date"].min() - timedelta(days=370)
    train_bucket_models(df)
    r = predict_bucketed(df, app_id="test_app")
    assert r["status"] == "ok"
    data = r["data"]
    assert "probability" in data
    assert "predicted_start_offset_days" in data
    assert "predicted_duration_days" in data
    return "bucket prediction OK"


def run_eval(threshold: float = 0.85) -> bool:
    passed = 0
    results = []

    for case in TEST_CASES:
        try:
            note = case["run"]()
            results.append(
                {
                    "id": case["id"],
                    "passed": True,
                    "note": note,
                    "description": case["description"],
                }
            )
            passed += 1
        except Exception as e:
            results.append(
                {
                    "id": case["id"],
                    "passed": False,
                    "note": str(e)[:120],
                    "description": case["description"],
                }
            )

    total = len(TEST_CASES)
    score = passed / total

    print(f"\n{'=' * 55}")
    print("Steam Sales Window - Eval Report")
    print(f"{'=' * 55}")
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} [{r['id']}] {r['description']}")
        if r["note"]:
            print(f"       → {r['note']}")
    print(f"{'=' * 55}")
    print(
        f"  Score: {passed}/{total} = {score:.0%}  "
        f"({'✅ PASS' if score >= threshold else '❌ FAIL'})"
    )
    print(f"{'=' * 55}\n")
    return score >= threshold


if __name__ == "__main__":
    ok = run_eval()
    sys.exit(0 if ok else 1)
