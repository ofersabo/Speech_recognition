import os
import pickle
from gcommand_loader import GCommandLoader
import torch
from matplotlib import pyplot as plt

train_dataset = GCommandLoader('./data/train/')
valid_dataset = GCommandLoader('./data/valid/')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

def plot_loss(acc, train_loss, dev_loss):
    preffix = '/tmp/SR_OE/'
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    plt_file_name = preffix + 'acc.png'
    plt.plot(acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = preffix + 'train_loss.png'
    plt.plot(train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = preffix + 'dev_loss.png'
    plt.plot(dev_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig(plt_file_name)
    plt.close()


def save_to_file(var, file):
    preffix = '/tmp/SR_OE/'
    out_file = open(file, "wb")
    pickle.dump(var, out_file)
    out_file.close()

    out_file = open(preffix + file, "wb")
    pickle.dump(var, out_file)
    out_file.close()


def load_from_file(file):
    in_file = open(file, "rb")
    var = pickle.load(in_file)
    in_file.close()
    return var


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")