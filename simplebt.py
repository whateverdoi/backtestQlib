import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
import pandas as pd

# 1. 初始化 (确保路径和你下载时一致)
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 2. 核心配置：数据集与模型
# 我们使用 Qlib 内置的 Alpha158 指子集和 LightGBM 模型
market = "csi300"  # 股票池：沪深300
benchmark = "SH000300" # 基准

data_handler_config = {
    "start_time": "2018-01-01",
    "end_time": "2020-12-31",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2019-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.88,
            "learning_rate": 0.04,
            "subsample": 0.87,
            "n_estimators": 200,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2018-01-01", "2019-12-31"),
                "valid": ("2020-01-01", "2020-05-30"),
                "test": ("2020-06-01", "2020-12-31"),
            },
        },
    },
}

# 3. 训练模型
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

with R.start(experiment_name="workflow"):
    model.fit(dataset)
    # 预测
    pred_score = model.predict(dataset)
    print("预测得分结果 (前5行):")
    print(pred_score.head())

# 4. 简单分析 (直接使用 dataset 接口)
from qlib.contrib.evaluate import backtest_daily

# 获取测试集的 Label 用于对比
# 'test' 对应你在 task 字典里定义的 segments
label = dataset.prepare("test", col_set="label")

# 确保预测值和标签的索引一致
# pred_score 的 index 通常是 [datetime, instrument]
pred_score = pred_score.reindex(label.index)

# 计算每日 IC
def calculate_ic(df):
    # 第一列是预测分，第二列是 label
    return df.iloc[:, 0].corr(df.iloc[:, 1])

combined = pd.concat([pred_score, label], axis=1)
daily_ic = combined.groupby(level='datetime').apply(calculate_ic)

print(f"\n测试集平均 IC 值: {daily_ic.mean():.4f}")
print(f"IC 标准差: {daily_ic.std():.4f}")