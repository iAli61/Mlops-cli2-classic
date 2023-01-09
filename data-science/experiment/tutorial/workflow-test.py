

# %% [markdown]
# # Introduction
# Though users can automatically run the whole Quant research worklfow based on configurations with Qlib.
# 
# Some advanced users usally would like to carefully customize each component to explore more in Quant.
# 
# If you just want a simple example of Qlib. [Quick start](https://github.com/microsoft/qlib#quick-start) and [workflow_by_code](https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.ipynb) may be a better choice for you.
# 
# If you want to know more details about Quant research, this notebook may be a better place for you to start.
# 
# We hope this script could be a tutorial for users who are interested in the details of Quant.
# 
# This notebook tries to demonstrate how can we use Qlib to build components step by step. 

# %%
from pprint import pprint
from pathlib import Path
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.report import analysis_model, analysis_position

# %%
import azureml.core
from azureml.core import Workspace
import mlflow

ws = Workspace.from_config()

print("SDK version:", azureml.core.VERSION)
print("MLflow version:", mlflow.version.VERSION)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# %%
MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"

# %%
from qlib.tests.data import GetData
GetData().qlib_data(exists_skip=True, target_dir="~/.qlib/qlib_data/cn_data", delete_old=False)

# %%
import qlib
from qlib.config import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# %% 
handler_kwargs = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": MARKET,
}
handler_conf = {
    "class": "Alpha158",
    "module_path": "qlib.contrib.data.handler",
    "kwargs": handler_kwargs,
}
pprint(handler_conf)
hd = init_instance_by_config(handler_conf)

# %%
dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": hd,
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
}
dataset = init_instance_by_config(dataset_conf)

# %%
model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
})

# %% [markdown]
# # Evaluation:
# - Signal-based
# - Portfolio-based: backtest 

# %%
###################################
# prediction, backtest & analysis
###################################
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}


# %%
# start exp to train model
with R.start(experiment_name=EXP_NAME, uri=ws.get_mlflow_tracking_uri()):
    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()

    sar = SigAnaRecord(rec)
    sar.generate()
    
    #  portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config, "day")
    par.generate()


# %%
from mlflow.tracking import MlflowClient

# Use MlFlow to retrieve the run that was just completed
client = MlflowClient()
run_id = rid
finished_mlflow_run = MlflowClient().get_run(run_id)

metrics = finished_mlflow_run.data.metrics
tags = finished_mlflow_run.data.tags
params = finished_mlflow_run.data.params
artifact_list = client.list_artifacts(run_id)

print(metrics,tags,params,artifact_list,sep='\n')



# %%
from qlib.workflow.recorder import MLflowRecorder
recorder = MLflowRecorder(name=EXP_NAME, experiment_id=run_id, uri=ws.get_mlflow_tracking_uri(), mlflow_run=finished_mlflow_run)

# load previous results
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# %%
analysis_df.head()

# %%
# Previous Model can be loaded. but it is not used.
loaded_model = recorder.load_object("trained_model")
loaded_model

# %%

def log_figure(figs, name, height=600, width=1200):
    for i in range(len(figs)):
        figs[i].update_layout(
            margin=dict(l=20, r=20, t=50, b=80),
            paper_bgcolor="LightSteelBlue",
            height=height, 
            width=width,
        )
        # resume mlflow run and log the figure
        recorder.client.log_figure(run_id, figs[i], f"{name}-{i}.png")

# %% [markdown]
# ## analysis position

### report



# %%
# analysis_position.report_graph(report_normal_df, show_notebook=False)
figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
log_figure(figs, "report", height=1200, width=1200)

# %% [markdown]
# ### risk analysis

# %%
figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
log_figure(figs, "risk_analysis")

# %% [markdown]
# ## analysis model

# %%
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ['label']
# score IC
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
log_figure(figs, "score_ic")
# model performance
figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
log_figure(figs, "model_performance")


