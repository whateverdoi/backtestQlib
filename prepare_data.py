import pandas as pd
import numpy as np
import talib
from pathlib import Path
from config import RAW_DATA, QLIB_DATA_DIR, TRIPLE_BARRIER


def compute_triple_barrier(close, high, low, atr, atr_multiple=2.0, vertical_bars=30):
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    start = 20
    end = n - vertical_bars - 1

    for i in range(start, end):
        upper = close[i] + atr_multiple * atr[i]
        lower = close[i] - atr_multiple * atr[i]
        hit = 0
        for j in range(1, vertical_bars + 1):
            idx = i + j
            if high[idx] >= upper:
                hit = 1; break
            if low[idx] <= lower:
                hit = -1; break
        labels[i] = hit

    return labels


def write_feature_bin(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.hstack([[0], np.asarray(values, dtype=np.float32)]).astype("<f")
    with open(path, "wb") as fp:
        data.tofile(fp)


def build_qlib_bin(df, labels):
    qlib_dir = QLIB_DATA_DIR

    cal_path = qlib_dir / "calendars" / "day.txt"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text("\n".join(df["date"].astype(str)) + "\n")
    print(f"Calendar: {cal_path} ({len(df)} entries)")

    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    line = f"SOLUSDT\t{df['date'].iloc[0]}\t{df['date'].iloc[-1]}\n"
    for name in ["all", "solusdt"]:
        (inst_dir / f"{name}.txt").write_text(line)
    print(f"Instruments: {inst_dir}")

    feat_dir = qlib_dir / "features" / "solusdt"
    feat_dir.mkdir(parents=True, exist_ok=True)

    fields = {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
        "vwap": ((df["high"] + df["low"] + df["close"]) / 3).values,
        "label": labels.astype(np.float32),
    }

    for field, values in fields.items():
        path = feat_dir / f"{field}.day.bin"
        write_feature_bin(path, values)
        print(f"  {field}: {path}")

    print(f"\nQlib bin data at: {qlib_dir.resolve()}")
    dist = np.bincount(labels.astype(int) + 1)
    print(f"Label distribution [-1, 0, 1]: {dist}")


def main():
    QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA)
    df["date"] = df["timestamp"]

    atr = talib.ATR(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float), timeperiod=20)
    atr = np.nan_to_num(atr, nan=0.0)

    labels = compute_triple_barrier(
        df["close"].values, df["high"].values, df["low"].values,
        atr,
        TRIPLE_BARRIER["atr_multiple"],
        TRIPLE_BARRIER["vertical_bars"],
    )

    build_qlib_bin(df, labels)


if __name__ == "__main__":
    main()
