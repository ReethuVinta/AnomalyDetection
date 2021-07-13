import pickle
import warnings
from datetime import datetime
import os
from typing import Callable, Dict, Iterable
import argparse
import sys
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
from avalanche.training.strategies import GEM,EWC 
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import sys

warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="GEM",
        help="The type (EWC|GEM) of the model architecture",
    )
    args = parser.parse_args(argv)
    return args


args = get_args(sys.argv[1:])

# Creating folder to save weights
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

cache_label = "./data/train_data_x.pth"

if os.path.exists(cache_label):

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

    data_path = "../data/dataset.npy"
    labels_path = "../data/labels.npy"
    ds = np.load(data_path)
    label = np.load(labels_path)

    whole = np.concatenate((ds, label), axis=1)

    df = pd.DataFrame(whole, columns=[str(i) for i in range(whole.shape[1])])

    y = df.pop(df.columns[-1]).to_frame() - 1

    # Normalsing Dataset
    X = (df - df.min()) / (df.max() - df.min() + 1e-5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.33
    )

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    # Labelling classses as 0 or 1 based on type of class.
    for i in range(y_train.iloc[:, -1].nunique()):
        train_dict["cat" + str(i)] = X_train[y_train.iloc[:, -1] == i]

        temp = y_train[y_train.iloc[:, -1] == i]

        if i == 0:
            temp.loc[:, "70"] = 0
        else:
            temp.loc[:, "70"] = 1
        train_label_dict["cat" + str(i)] = temp

    for i in range(y_test.iloc[:, -1].nunique()):
        test_dict["cat" + str(i)] = X_test[y_test.iloc[:, -1] == i]

        temp = y_test[y_test.iloc[:, -1] == i]

        if i == 0:
            temp.loc[:, "70"] = 0
        else:
            temp.loc[:, "70"] = 1
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


class Conv(nn.Module):
    def __init__(self, num_classes=2, input_size=70, hidden_size=100):
        super().__init__()

        self.conv1d = nn.Conv1d(1, 14, kernel_size=3)

        self.feature1 = nn.Sequential(
            nn.Linear(14 * 68, hidden_size), nn.ReLU(inplace=True), nn.Dropout()
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.view(x.size(0), 1, self._input_size)
        # (1,128,70)
        x = self.conv1d(x)
        x = x.view(-1, 14 * 68)
        x = self.feature1(x)
        x = self.classifier(x)
        return x


# Model Creation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Conv()
model.to(device)

# Model architecture
arch = args.model_architecture.upper()

# Task Orders
perm1 = {
    "1": [0, 1, 2],
    "2": [3, 4, 5],
    "3": [6, 7, 8],
    "4": [9, 10, 11],
    "5": [12, 13, 14],
}
perm2 = {
    "1": [2, 4, 8],
    "2": [5, 9, 0],
    "3": [12, 1, 7],
    "4": [3, 14, 13],
    "5": [10, 11, 6],
}
perm3 = {
    "1": [14, 9, 12],
    "2": [4, 3, 5],
    "3": [0, 11, 8],
    "4": [7, 6, 10],
    "5": [2, 13, 1],
}
perm4 = {
    "1": [1, 7, 4],
    "2": [3, 12, 2],
    "3": [10, 6, 11],
    "4": [13, 8, 0],
    "5": [9, 14, 5],
}
perm5 = {
    "1": [10, 13, 14],
    "2": [3, 5, 6],
    "3": [9, 4, 2],
    "4": [1, 12, 8],
    "5": [7, 11, 0],
}

task_order_list = [perm1, perm2, perm3, perm4, perm5]


for task_order in range(len(task_order_list)):
    print("Current task order processing : ", task_order + 1)
    dataset = task_ordering(task_order_list[task_order])

    generic_scenario = tensor_scenario(
        train_data_x=dataset[0],
        train_data_y=dataset[1],
        test_data_x=dataset[2],
        test_data_y=dataset[3],
        task_labels=[
            0 for key in task_order_list[i].keys()
        ],  # shouldn't provide task ID for inference
    )

    # log to Tensorboard
    tb_logger = TensorboardLogger(f"./tb_data/{cur_time}_CNN1D_0inTask{task_order}/")

    # log to text file
    text_logger = TextLogger(
        open(f"./logs/{cur_time}_CNN1D_0inTask{task_order}.txt", "w+")
    )

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

    if arch == "GEM":
        cl_strategy = GEM(
            model,
            optimizer=Adam(model.parameters()),
            patterns_per_exp=4400,
            criterion=CrossEntropyLoss(),
            train_mb_size=128,
            train_epochs=50,
            eval_mb_size=128,
            evaluator=eval_plugin,
            device=device,
        )
    else:
        cl_strategy = EWC(
            model,
            optimizer=Adam(model.parameters()),
            ewc_lambda=0.001,
            criterion=CrossEntropyLoss(),
            train_mb_size=128,
            train_epochs=50,
            eval_mb_size=128,
            evaluator=eval_plugin,
            device=device,
        )

    # TRAINING LOOP
    print("Starting experiment...")

    os.makedirs(f"CNN1D_0inTask{task_order+1}", exist_ok=True)

    results = []

    for task_number, experience in enumerate(generic_scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)

        print("Training completed!")

        torch.save(
            model.state_dict(),
            "./weights/CNN1D_0inTask{}/After_training_Task_{}".format(
                task_order + 1, task_number
            ),
        )
        print("Model saved!")
        print("Computing accuracy on the whole test set")
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(generic_scenario.test_stream))
