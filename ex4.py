import ntpath

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cer import *
from our_model import *
from utils import *

ctc_loss = torch.nn.CTCLoss(blank=0,reduction='mean',zero_infinity=True)
c2i = "_abcdefghijklmnopqrstuvwxyz"


def apply(model, ctc_loss_function, batch, labels):
    lengths = torch.LongTensor([model.get_seq_length(s.size(2)) for s in batch])
    target_lengths = torch.LongTensor([len(idx_to_class[label.item()]) for label in labels])
    targets = []
    max_length = target_lengths.max().item()
    for word_index in labels:
        word = idx_to_class[word_index.item()]
        chars = []
        for c in word:
            chars.append(c2i.index(c))
        target_word = torch.tensor(chars)  # padding when using batch
        pad_length = max_length - len(chars)
        padded_tensor = F.pad(target_word, (0, pad_length),value=-1)
        targets.append(padded_tensor)
    targets = torch.stack(targets).to(device)
    batch = batch.to(device)
    pred = model(batch)
    pred = pred.permute(1, 0, 2).to(device)
    loss = ctc_loss_function(pred, targets, lengths, target_lengths)
    return pred, loss



def greedy_decoding(pred):
    '''
    :param pred: batch of probability shape (101,100,27)
    :return: word
    '''
    list_of_words = []
    raw_words = []
    pred = pred.permute(1, 0, 2)
    for seq in pred:
        chars = torch.argmax(seq, dim=1).tolist()
        raw_word = "".join([c2i[ch] for ch in chars])
        word = [chars[0]] + [c for index, c in enumerate(chars[1:], start=1) if c != chars[index - 1]]
        word = [c2i[i] for i in word if i != 0]
        word = "".join(word)
        # print(word)
        list_of_words.append(word)
        raw_words.append(raw_word)
    return list_of_words, raw_words


def accuracy_on_dev(model, dev, epoch, print_to_screen = False,file_name = "dev_word_list"):
    loss_per_batch = []
    model.eval()
    acc_on_dev = 0
    total_cer = []
    for k, (batch_input, batch_label,batch_path) in enumerate(dev):
        pred, loss = apply(model, ctc_loss, batch_input, batch_label)
        loss_per_batch.append(loss.item())
        pred_words, pred_raw_words = greedy_decoding(pred)
        gold_list = [idx_to_class[word.item()] for word in batch_label]
        gold_list_files = [path for path in batch_path]
        gold_set = set([(idx_to_class[word.item()], word_index) for word_index, word in enumerate(batch_label)])
        pred_set = set([(word, word_index) for word_index, word in enumerate(pred_words)])
        acc_on_dev += len(pred_set.intersection(gold_set)) / len(pred_set)
        total_cer.append((np.array(list(map(lambda x: cer(x[0], x[1]), zip(pred_words, gold_list))))).mean())

    print_to_file(pred_words,gold_list, pred_raw_words, gold_list_files, epoch, print_to_screen,file_name=file_name)
    return total_cer, acc_on_dev / len(batch_input) , loss_per_batch


def predict_test(model, testdata, file_name = "dev_word_list"):
    model.eval()
    all_pred_words = []
    all_word_files = []
    for k, (batch_input, batch_path) in enumerate(testdata):
        batch = batch_input.to(device)
        pred = model(batch)
        pred = pred.permute(1, 0, 2).to(device)
        pred_words, pred_raw_words = greedy_decoding(pred)
        word_files = [ntpath.basename(path) for path in batch_path]
        all_pred_words.extend(pred_words)
        all_word_files.extend(word_files)

    print_test_file(all_pred_words, all_word_files,file_name=file_name)



def save_model(model_to_save):
    print("saved model")
    torch.save(model_to_save.state_dict(), PATH_to_model)


def train_model(model, train, dev):
    train_error_rate = []
    error_rate_graph = []
    loss_on_dev_per_epoch = []
    loss_on_train_per_epoch = []
    min_error_rate = 999
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    while min_error_rate > 1:
        for epoch in range(1000):
            loss_history_per_batch = []
            model.train()
            print("Epoch {}".format(epoch))
            for k, (batch_input, batch_label, batch_path) in enumerate(train):
                pred, loss = apply(model, ctc_loss, batch_input, batch_label)
                loss_history_per_batch.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            char_error_rate_list_per_batch, exact_acc_mean_on_dev, loss_on_dev = accuracy_on_dev(model, dev, epoch,print_to_screen=False)

            if epoch % 20 == 0:
                x = accuracy_on_dev(model, train , epoch,print_to_screen=False, file_name="train_word_list")
                train_error_rate.append(np.array(x[0]).mean())
                plot_error_rate_graph(train_error_rate,file_name="train_error_rate" + '.png')

            mean_error_rate = sum(char_error_rate_list_per_batch)/len(char_error_rate_list_per_batch)
            print("mean Error rate on last Epoch is ")
            print(mean_error_rate)

            plot_loss_inside_epoch(loss_history_per_batch, loss_on_dev)
            error_rate_graph.append(mean_error_rate)
            loss_on_dev_per_epoch.append(sum(loss_on_dev)/len(loss_on_dev))
            loss_on_train_per_epoch.append(sum(loss_history_per_batch)/len(loss_history_per_batch))

            plot_loss(loss_on_train_per_epoch, loss_on_dev_per_epoch, error_rate_graph )

            save_to_file([loss_on_train_per_epoch, loss_on_dev_per_epoch,error_rate_graph], "history.pickle")
            print("saved history to pickle")

            if len(char_error_rate_list_per_batch) > 20 and mean_error_rate < min_error_rate:
                min_error_rate = mean_error_rate
                save_model(model)

    return model


if __name__ == '__main__':
    train_data = train_loader
    dev_data = valid_loader
    temp_path = "model_to_be_saved_cuda:2_er6_8.pth"
    global idx_to_class
    idx_to_class = {v: k for k, v in train_data.dataset.class_to_idx.items()}
    speech_model = SR_model().to(device)
    if os.path.isfile(PATH_to_model):
        speech_model.load_state_dict(torch.load(PATH_to_model, map_location=device))

    speech_model.eval()
    print("using gpu: " + str(use_cuda))

    train_model(speech_model, train_data, dev_data)

    predict_test(speech_model,test_loader,"test_y")


