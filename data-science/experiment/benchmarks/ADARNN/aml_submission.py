# %%

import os
import sys
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import azureml.core
from azureml.core import Workspace, Experiment, Run, Dataset

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

# create a new datastor in the workspace 
datastore = ws.get_default_datastore()
print(datastore.name, datastore.datastore_type, datastore.account_name, datastore.container_name)



# %%

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command

from pathlib import Path
from azure.ai.ml.entities import Environment, BuildContext

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
print(ml_client)

# %%
datapath = str(Path.home().joinpath('.qlib/qlib_data/us_data'))
print(f'Using data path: {datapath}')




my_data = Data(
    path=datapath,
    type=AssetTypes.URI_FOLDER,
    description="qlib_us",
    name="qlib_us"
)
ml_client.data.create_or_update(my_data)

# %%


# create a custiom environment from conda yaml file
env = Environment(
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.3-gpu:7",
    conda_file="conda.yaml",
    name="qlib_env",
    description="my custom environment",
)

env = ml_client.environments.create_or_update(env)

# %%
from azure.ai.ml import command
from azure.ai.ml import Input

inputs = {
    "input_data": Input(type=AssetTypes.URI_FOLDER, path="azureml:qlib_cn:1")
}

job = command(
    code=".",  # local path where the code is stored
    command="python ./workflow.py --experiment_name ADARNN --config_file workflow_config_adarnn_Alpha360.yaml --input_data ${{inputs.input_data}}",
    inputs=inputs,
    environment="qlib_env@latest",
    compute="gup-teslaM60",
)

# submit the command
returned_job = ml_client.jobs.create_or_update(job)
# get a URL for the status of the job
returned_job.services["Studio"].endpoint
# %%
