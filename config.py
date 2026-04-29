from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "dollarbars.csv"
PROCESSED_DATA = DATA_DIR / "processed" / "data.pkl"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"

TRIPLE_BARRIER = {
    "atr_period": 20,
    "atr_multiple": 2.0,
    "vertical_bars": 30,
}

SPLIT = {
    "train": 0.7,
    "valid": 0.15,
    "test": 0.15,
}

COSTS = {
    "fee": 0.001,
    "slippage": 0.0005,
}

SIGNAL_PERCENTILE = 30

MODELS = {
    "lgbm": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.88,
            "learning_rate": 0.05,
            "subsample": 0.87,
            "n_estimators": 200,
            "early_stopping_rounds": 50,
        },
    },
    "ridge": {
        "class": "LinearModel",
        "module_path": "qlib.contrib.model.linear",
        "kwargs": {
            "estimator": "ridge",
            "alpha": 1.0,
            "fit_intercept": False,
        },
    },
    "xgb": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "n_estimators": 200,
            "early_stopping_rounds": 50,
        },
    },
}
