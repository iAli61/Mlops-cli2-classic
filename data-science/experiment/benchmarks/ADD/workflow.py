#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces. 
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import qlib

from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

import pandas as pd

import azureml.core
from azureml.core import Workspace
import mlflow


from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

# read yaml file Users/alibina/notebooks/learning/qlib/examples/portfolio/config_enhanced_indexing.yaml
import yaml
from pathlib import Path

import argparse

# write a funtion to parse input arguments: experiment name, config file
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    return args

# replace the matching value in the dict recursively
def replace_item(obj, value, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, value, replace_value)
        elif v == value:
            obj[k] = replace_value
    return obj


def log_figure(recorder, figs, name, height=600, width=1200):
    for i in range(len(figs)):
        figs[i].update_layout(
            margin=dict(l=20, r=20, t=50, b=80),
            paper_bgcolor="LightSteelBlue",
            height=height, 
            width=width,
        )
        # resume mlflow run and log the figure
        recorder.client.log_figure(recorder.id, figs[i], f"{name}-{i}.png")

def logs_analysis(recorder):
    # Use MlFlow to retrieve the run that was just completed
    
    # load previous results
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    # analysis_position.report_graph(report_normal_df, show_notebook=False)
    figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
    log_figure(recorder, figs, "report", height=1200, width=1200)

    figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    log_figure(recorder, figs, "risk_analysis")

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    # score IC
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    log_figure(recorder, figs, "score_ic")
    # model performance
    figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    log_figure(recorder, figs, "model_performance")

def main(config, experiment_name):
    # use default data
    GetData().qlib_data(target_dir=config['qlib_init']['provider_uri'], region=config['qlib_init']['region'], exists_skip=True)
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=config['qlib_init']['region'])

    model = init_instance_by_config(config['model'])
    dataset = init_instance_by_config(config['dataset'])

    port_analysis_config = config["port_analysis_config"]
    port_analysis_config = replace_item(port_analysis_config, "<MODEL>", model)
    port_analysis_config = replace_item(port_analysis_config, "<DATASET>", dataset)

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    ws = Workspace.from_config()

    print("SDK version:", azureml.core.VERSION)
    print("MLflow version:", mlflow.version.VERSION)
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
    
    # start exp
    with R.start(experiment_name=experiment_name, uri=ws.get_mlflow_tracking_uri()):
        rec = R.get_recorder()
        rid = rec.id
        print(f"recorder id: {rid}")
        # log parameters
        for key, value in config['model'].items():
            print(f"{key}: {value}")
            rec.client.log_param(run_id=rid, key=f'{key}-model', value=value)
        for key, value in config['dataset'].items():
            print(f"{key}: {value}")
            rec.client.log_param(run_id=rid, key=f'{key}-dataset', value=value)
        # training
        model.fit(dataset)
        R.save_objects(trained_model=model)
        

        # prediction
        sr = SignalRecord(model, dataset, rec)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(rec)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(rec, port_analysis_config, "day")
        par.generate()

        # plot results
        logs_analysis(recorder=rec)

if __name__ == "__main__":
    
    args = parse_args()
    print(f'configpath: {args.config_file}')
    with open(args.config_file, "r") as fp:
        config = yaml.safe_load(fp)
        
    main(config, args.experiment_name)