import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from qlib.data import D
import pandas as pd
import numpy as np
from config import MODELS, SPLIT, COSTS, SIGNAL_PERCENTILE, RAW_DATA, QLIB_DATA_DIR, PROCESSED_DATA
from handler.crypto_handler import CryptoAlphaHandler


def _get_segments():
    cal = D.calendar(freq="day")
    n = len(cal)
    t1 = int(n * SPLIT["train"])
    t2 = int(n * (SPLIT["train"] + SPLIT["valid"]))
    return {
        "train": (str(cal[0]), str(cal[t1])),
        "valid": (str(cal[t1]), str(cal[t2])),
        "test": (str(cal[t2]), str(cal[-1])),
    }


def _dataset_config(segments):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "CryptoAlphaHandler",
                "module_path": "handler.crypto_handler",
                "kwargs": {
                    "instruments": "SOLUSDT",
                    "start_time": segments["train"][0],
                    "end_time": segments["test"][1],
                    "fit_start_time": segments["train"][0],
                    "fit_end_time": segments["valid"][1],
                },
            },
            "segments": segments,
        },
    }


def get_test_prices(segments):
    raw = pd.read_csv(RAW_DATA)
    raw["date"] = pd.to_datetime(raw["timestamp"])
    price_map = raw.set_index("date")["close"]
    test_dates = D.calendar(freq="day")[int(len(D.calendar(freq="day")) * (1 - SPLIT["test"])):]
    return price_map.reindex(test_dates).ffill().values


def backtest(pred_values, price_values, costs, percentile):
    n = len(pred_values)
    lo = np.percentile(pred_values, percentile)
    hi = np.percentile(pred_values, 100 - percentile)
    cash, shares, pos = 1.0, 0.0, 0
    eq = [1.0]
    trades = 0

    for t in range(n - 1):
        new_pos = 1 if pred_values[t] > hi else (-1 if pred_values[t] < lo else 0)
        if new_pos != pos:
            price = price_values[t + 1]
            cost = price * (costs["fee"] + costs["slippage"])
            if pos != 0:
                cash += shares * price - abs(shares) * cost
                shares = 0.0
            if new_pos != 0:
                shares = (cash / price) * new_pos
                cash = 0.0
            pos = new_pos
            trades += 1
        eq.append(cash + shares * price_values[t + 1])

    eq = pd.Series(eq)
    ret = eq.pct_change().dropna()
    return eq, ret, trades


def compute_metrics(eq, ret):
    tr = eq.iloc[-1] / eq.iloc[0] - 1
    sr = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else 0.0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    wr = (ret > 0).mean()
    return {"Return": f"{tr:.2%}", "Sharpe": f"{sr:.3f}", "MaxDD": f"{mdd:.2%}", "WinRate": f"{wr:.2%}"}


def main():
    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)

    segments = _get_segments()
    print("Split segments:")
    for k, v in segments.items():
        print(f"  {k}: {v[0][:19]} -> {v[1][:19]}")

    dataset_config = _dataset_config(segments)
    dataset = init_instance_by_config(dataset_config)
    print(f"\nHandler: {type(dataset.handler).__module__}.{type(dataset.handler).__name__}")
    train_f = dataset.prepare("train", col_set="feature")
    print(f"Alpha158 features: {train_f.shape[1]} columns, e.g. {train_f.columns[:5].tolist()}")
    print(f"Train samples: {train_f.shape[0]}")

    test_prices = get_test_prices(segments)

    all_results = {}

    for name, mc in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print('='*55)

        model = init_instance_by_config(mc)
        with R.start(experiment_name=f"bt_{name}"):
            model.fit(dataset)
            pred = model.predict(dataset)
            pv = pred.values.flatten()

            if len(pv) < 10:
                print(f"  Insufficient test samples: {len(pv)}"); continue

            if name == "ridge":
                print(f"  Ridge pred stats: min={pv.min():.4f} max={pv.max():.4f} "
                      f"mean={pv.mean():.4f} std={pv.std():.4f}")

            eq, rt, trades = backtest(pv, test_prices[:len(pv)+1], COSTS, SIGNAL_PERCENTILE)
            metrics = compute_metrics(eq, rt)
            metrics["Trades"] = str(trades)
            all_results[name] = metrics

            for k, v in metrics.items():
                print(f"  {k}: {v}")

    print(f"\n{'='*55}")
    print("  Model Comparison")
    print('='*55)
    print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<8}".format("Model", "Return", "Sharpe", "MaxDD", "WinRate", "Trades"))
    print('-'*56)
    for name, m in all_results.items():
        print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
            name, m["Return"], m["Sharpe"], m["MaxDD"], m["WinRate"], m.get("Trades", "-")))


if __name__ == "__main__":
    main()
