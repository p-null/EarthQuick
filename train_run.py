import os
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.train.estimator import Estimator

# Create workspace

workspace = Workspace.from_config()


# Create experiment

experiment_name = "test"
experiment = Experiment(workspace=workspace, name=experiment_name)


# Create/Attact existing compute resource

cluster_type = os.environ.get("AML_COMPUTE_CLUSTER_TYPE", "CPU")
compute_target = workspace.get_default_compute_target(cluster_type)


# Prepare data

data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

import urllib.request
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'train-images.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'train-labels.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'test-images.gz'))
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'test-labels.gz'))


# Upload data to cloud

data_store = workspace.get_default_datastore()
data_store.upload(src_dir=data_folder, target_path="target_data_folder", overwrite=True, show_progress=True)


# Crate directory & training script

script_folder = os.path.join(os.getcwd(), "script_folder")
os.makedirs(script_folder, exist_ok=True)


# Create a estimator


script_params = {
    '--data-folder': data_store.path('target_data_folder').as_mount(),
    '--regularization': 0.05
}

estimator = Estimator(source_directory=script_folder,
                        script_params=script_params,
                        compute_target=compute_target,
                        entry_script='train.py',
                        conda_packages=['scikit-learn'])


# Submit job to cluster

run = experiment.submit(config=estimator)


# Register model

model = run.register_model(model_name="xgb_earth", model_path="outputs/xgb_model.pkl")


# 