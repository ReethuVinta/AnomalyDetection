import argparse
import os
import pickle
import warnings
from datetime import datetime

import avalanche
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from avalanche.benchmarks.generators import tensor_scenario
from avalanche.evaluation.metrics import (
    ExperienceForgetting,
    StreamConfusionMatrix,
    accuracy_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import GEM
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, Subset
from torchsummary import summary
from torchvision import transforms

warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

# Creating folder to save weights
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

cache_label = "./data/train_data_x.pth"
# Normal = 0
# Attack = 1
if path.exists(cache_label):

    with open("./data/train_data_x.pth", "rb") as f:
        train_data_x = pickle.load(f)
    with open("./data/train_data_y.pth", "rb") as f:
        train_data_y = pickle.load(f)

    with open("./data/test_data_x.pth", "rb") as f:
        test_data_x = pickle.load(f)

    with open("./data/test_data_y.pth", "rb") as f:
        test_data_y = pickle.load(f)
    print("Data Loaded!!!")

else:
    with open("./data/kdd_df.pth", "rb") as f:
        df = pickle.load(f)

    y = df.pop(df.columns[-1]).to_frame()

    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.33
    )

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    for i in range(y_train.iloc[:, -1].nunique()):
        train_dict["cat" + str(i)] = X_train[y_train.iloc[:, -1] == i]

        temp = y_train[y_train.iloc[:, -1] == i]

        # Class label 11 = Normal class
        if i == 11:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1
        train_label_dict["cat" + str(i)] = temp

    for i in range(y_test.iloc[:, -1].nunique()):
        test_dict["cat" + str(i)] = X_test[y_test.iloc[:, -1] == i]

        temp = y_test[y_test.iloc[:, -1] == i]

        if i == 11:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1
        test_label_dict["cat" + str(i)] = temp

    train_data_x = list(torch.Tensor(train_dict[key].to_numpy()) for key in train_dict)
    train_data_y = list(
        torch.Tensor(train_label_dict[key].to_numpy()) for key in train_label_dict
    )
    test_data_x = list(torch.Tensor(test_dict[key].to_numpy()) for key in test_dict)
    test_data_y = list(
        torch.Tensor(test_label_dict[key].to_numpy()) for key in test_label_dict
    )

    with open("./data/train_data_x.pth", "wb") as f:
        pickle.dump(train_data_x, f)

    with open("./data/train_data_y.pth", "wb") as f:
        pickle.dump(train_data_y, f)

    with open("./data/test_data_x.pth", "wb") as f:
        pickle.dump(test_data_x, f)

    with open("./data/test_data_y.pth", "wb") as f:
        pickle.dump(test_data_y, f)

    print("Dumped into ./data/")


def task_ordering(perm):
    """Divides Data into tasks based on the given permutation order

    Parameters
    ----------
    perm : dict
        Dictionary containing task id and the classes present in it.

    Returns
    -------
    tuple
        Final dataset divided into tasks
    """
    final_train_data_x = []
    final_train_data_y = []
    final_test_data_x = []
    final_test_data_y = []

    for key, values in perm.items():
        temp_train_data_x = torch.Tensor([])
        temp_train_data_y = torch.Tensor([])
        temp_test_data_x = torch.Tensor([])
        temp_test_data_y = torch.Tensor([])

        for value in values:
            temp_train_data_x = torch.cat([temp_train_data_x, train_data_x[value]])
            temp_train_data_y = torch.cat([temp_train_data_y, train_data_y[value]])
            temp_test_data_x = torch.cat([temp_test_data_x, test_data_x[value]])
            temp_test_data_y = torch.cat([temp_test_data_y, test_data_y[value]])

        final_train_data_x.append(temp_train_data_x)
        final_train_data_y.append(temp_train_data_y)
        final_test_data_x.append(temp_test_data_x)
        final_test_data_y.append(temp_test_data_y)

    return final_train_data_x, final_train_data_y, final_test_data_x, final_test_data_y


# MODEL CREATION
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(num_classes=2, input_size=38, hidden_size=100)
model.to(device)

perm = {
    "1": [13, 22, 20, 14, 6],
    "2": [9, 10, 0, 1, 2],
    "3": [11, 15, 17, 21],
    "4": [18, 19, 7, 8, 12],
    "5": [3, 4, 5, 16],
}

task_order_list = [perm1]

dataset = task_ordering(task_order_list[task_order])

generic_scenario = tensor_scenario(
    train_data_x=dataset[0],
    train_data_y=dataset[1],
    test_data_x=dataset[2],
    test_data_y=dataset[3],
    task_labels=[
        0 for key in task_order_list[task_order].keys()
    ],  # shouldn't provide task ID for inference
)

# log to Tensorboard
tb_logger = TensorboardLogger(f"./tb_data/{cur_time}-SimpleMLP/")

# log to text file
text_logger = TextLogger(open(f"./logs/{cur_time}-SimpleMLP.txt", "w+"))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    ExperienceForgetting(),
    cpu_usage_metrics(experience=True),
    StreamConfusionMatrix(num_classes=2, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger],
)

cl_strategy = GEM(
    model,
    optimizer=Adam(model.parameters()),
    patterns_per_exp=1470,
    criterion=CrossEntropyLoss(),
    train_mb_size=128,
    train_epochs=50,
    eval_mb_size=128,
    evaluator=eval_plugin,
    device=device,
)

# TRAINING LOOP
print("Starting experiment...")

os.makedirs(os.path.join("weights", f"SimpleMLP", exist_ok=True))

results = []
i = 1
for task_number, experience in enumerate(generic_scenario.train_stream):
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)

    print("Training completed")
    torch.save(
        model.state_dict(),
        "./weights/SimpleMLP/After_training_Task_{}".format(task_number + 1),
    )
    print("Model saved!")
    print("Computing accuracy on the whole test set")
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(generic_scenario.test_stream))
