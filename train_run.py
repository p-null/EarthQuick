import os
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.train.estimator import Estimator
from azureml.core.compute import AmlCompute, ComputeTarget 

# Create workspace

workspace = Workspace.from_config()


# Create experiment

experiment_name = "test"
experiment = Experiment(workspace=workspace, name=experiment_name)


# Create/Attact existing compute resource

#compute_cluster_type = os.environ.get("AML_COMPUTE_compute_cluster_type", "CPU")
#compute_target = workspace.get_default_compute_target(compute_cluster_type)

compute_cluster_name = os.environ.get("AML_COMPUTE_compute_cluster_name", "cpu_cluster")
cluster_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
cluster_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D1_V2")

if compute_cluster_name in workspace.compute_targets:
    compute_target = workspace.compute_targets['compute_cluster_name']
    if compute_target and type(compute_target) is AmlCompute:
        print("using existing compute_target: {}".format(compute_target_name))
else:
    print("creating a new one")
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=cluster_min_nodes,
                                                                max_nodes=cluster_max_nodes)
    compute_target = ComputeTarget.create(workspace, compute_cluster_name, provisioning_config)


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