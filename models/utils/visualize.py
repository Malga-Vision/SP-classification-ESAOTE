import os
import torch
import pandas as pd
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pickle


def plot_confusion(predictions: list, targets: list, num_classes: int, class_name: list = [],
                   log_dir: str or None = None, title: str or None = None, format: str = 'png'): # type: ignore
    if not title:
        title = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    matrix = confmat(torch.Tensor(predictions), torch.Tensor(targets)).double()

    for i in range(num_classes):
        # M[i,j] stands for element of real class i that was classified as j
        sum = torch.sum(matrix[i, :])
        matrix[i, :] = matrix[i, :] / sum

    if len(class_name) > 0:
        label_list = class_name
    else:
        label_list = range(num_classes)
    df_cm = pd.DataFrame(matrix, label_list, label_list)

    annot_pd = df_cm.map(lambda x: "{:.2%}".format(x) if round(x, 3) != 0.000 else '0.00')

    mean_acc = torch.diag(matrix).sum() / num_classes

    plt.figure(figsize=(10, 7))
    sns.set_theme(font_scale=0.8)
    sns.heatmap(df_cm, annot=annot_pd, annot_kws={"size": 15}, fmt='s', vmin=0, vmax=1, cmap="Blues", cbar=False)
    plt.title("Mean acc: {:.2%}".format(mean_acc), fontsize=20)
    plt.xlabel('predicted label', )
    plt.ylabel('actual label')
    if log_dir:
        log_subdir = os.path.join(log_dir, "plots")
        os.makedirs(log_subdir, exist_ok=True)
        path = os.path.join(log_subdir, "confusion_{}.{}".format(title, format))
        plt.savefig(path, format=format)
    plt.show()


def plot_history(file_path, log_dir = None, fname_suffix: str or None = None, format: str = 'png'): # type: ignore
    if not fname_suffix:
        fname_suffix = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    with open(file_path, 'rb') as file:
        history = pickle.load(file)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
  
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    
    if log_dir:
        log_subdir = os.path.join(log_dir, "plots")
        if not os.path.exists(log_subdir):
            os.makedirs(log_subdir, exist_ok=True)
        path = os.path.join(log_subdir, "history_{}.{}".format(fname_suffix, format))
        plt.savefig(path, format=format)
    
    plt.show()
