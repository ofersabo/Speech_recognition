import os
import pickle
from gcommand_loader import GCommandLoader
import torch
from matplotlib import pyplot as plt
import random

random.seed(1)

train_dataset = GCommandLoader('./data/train/')
valid_dataset = GCommandLoader('./data/valid/')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
preffix = '/tmp/SR_OE/'


def print_to_file(list_of_words, gold_list,print_to_screen):
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    words_file_name = preffix + 'word_list.txt'
    with open(words_file_name, "w") as f:
        for p, g in zip(list_of_words, gold_list):
            f.write(p + "\t" + g + "\n")
    if print_to_screen:
        for p, g in zip(list_of_words, gold_list):
            print(p + "\t" + g)


def plot_loss_inside_epoch(loss_history_per_batch, loss_on_dev):
    title = "loss_inside_epoch.png"
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    plt_file_name = preffix + title
    plt.plot(loss_history_per_batch, 'r', label='train')
    plt.plot(loss_on_dev, 'g', label='dev')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(plt_file_name)
    plt.close()


def plot_loss(train_loss, dev_loss, error_rate_graph):
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    plt_file_name = preffix + 'error_rate.png'
    plt.plot(error_rate_graph)
    plt.xlabel('Iterations')
    plt.ylabel('error_rate')
    plt.title('error_rate')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = preffix + 'train_loss.png'
    plt.plot(train_loss, 'r', label='train')
    plt.plot(dev_loss, 'g', label='dev')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss on Epochs')
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

train_indices =  random.sample(range(30000), 5000)
test_indices = random.sample(range(6798), 500)
# print(train_indices)
# print(test_indices)
train_subset = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
dev_subset = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(test_indices))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
# device = torch.device("cuda:2")
