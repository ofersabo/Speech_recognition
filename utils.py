import os
import pickle
import random
from time import gmtime, strftime

import torch
from matplotlib import pyplot as plt

from gcommand_loader import GCommandLoader, GTestCommandLoader

random.seed(1)

train_dataset = GCommandLoader('./data/train/')
valid_dataset = GCommandLoader('./data/valid/')
test_dataset = GTestCommandLoader('./data/test/')


curr_time = strftime("%Y-%m-%d_%H%M%S", gmtime())


batch_size = 50
cuda_device = "cuda:2"
preffix = '/tmp/SR_OE' + curr_time + '/'
PATH_to_model = "model_to_be_saved_" + curr_time + ".pth"


def print_test_file(pred_words, word_files, file_name):
    if not os.path.exists(preffix):
            os.makedirs(preffix)
    test_file_name = preffix + file_name
    with open(test_file_name, "w") as f:
        for i in range(len(pred_words)):
            f.write("%s, %s\n" % (word_files[i], pred_words[i]))


def print_to_file(list_of_words, gold_list, pred_raw, files, epoch, print_to_screen,file_name = 'word_list.'):
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    words_file_name = preffix + file_name + str(epoch)+'.txt'
    with open(words_file_name, "w") as f:
        for i, gold in enumerate(gold_list):
            f.write("raw: %s\tpred: %s\tgold:%s\t file:%s\n" %
                    (pred_raw[i], list_of_words[i], gold, files[i]))

    if print_to_screen:
        for i, gold in enumerate(gold_list):
            print("raw: %s\tpred: %s\tgold:%s\t file:%s\n" %
                    (pred_raw[i], list_of_words[i], gold, files[i]))


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

def plot_error_rate_graph(error_rate,file_name =  'error_rate.png'):
    plt_file_name = preffix + file_name
    plt.plot(error_rate)
    plt.xlabel('Iterations')
    plt.ylabel('error_rate')
    plt.title('error_rate')
    plt.savefig(plt_file_name)
    plt.close()

def plot_loss(train_loss, dev_loss, error_rate_graph):
    if not os.path.exists(preffix):
        os.makedirs(preffix)
    plot_error_rate_graph(error_rate_graph)

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

def plot_raw_data(sample, word):
    plt.title(word)
    plt.imshow(sample.squeeze().numpy(), origin='lower')
    plt.show()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=20, pin_memory=True, sampler=None)


use_cuda = torch.cuda.is_available()
device = torch.device(cuda_device if use_cuda else "cpu")
