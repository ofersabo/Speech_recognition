from our_model import *
import torch.nn as nn
from our_model import get_seq_length
import numpy as np
from utils import *
from cer import *

ctc_loss = torch.nn.CTCLoss(blank=26)
c2i = "abcdefghijklmnopqrstuvwxyz"
PATH = "modelto_be_saved.pth"

speech_model = our_model().to(device)
loss_history = []
dev_loss = []
error_rate = []


def apply(model, ctc_loss, batch, labels):
    import torch.nn.functional as F
    lengths = torch.LongTensor([get_seq_length(s.size(2)) for s in batch])
    target_lengths = torch.LongTensor([len(idx_to_class[label.item()]) for label in labels])
    targets = []
    max_length = target_lengths.max()
    for word_index in labels:
        word = idx_to_class[word_index.item()]
        chars = []
        for c in word:
            chars.append(c2i.index(c))
        target_word = torch.tensor(chars)  # TODO  add padding when using batch
        pad_length = max_length - len(chars)
        padded_tensor = F.pad(target_word, (0, pad_length))
        targets.append(padded_tensor)
    targets = torch.stack(targets)
    batch = batch.to(device)
    pred = model(torch.autograd.Variable(batch))
    pred = pred.permute(1, 0, 2)
    loss = ctc_loss(pred, targets, lengths, target_lengths)
    return pred, loss



def greedy_decoding(pred):
    '''
    :param pred: batch of probability shape (101,100,27)
    :return: word
    '''
    list_of_words = []
    pred = pred.permute(1, 0, 2)
    for seq in pred:
        chars = torch.argmax(seq, dim=1).tolist()
        word = [chars[0]] + [c for index, c in enumerate(chars[1:], start=1) if c != chars[index - 1]]
        word = [c2i[i] for i in word if i != 26]
        word = "".join(word)
        # print(word)
        list_of_words.append(word)
    return list_of_words


def accuracy_on_dev(model, dev):
    model.eval()
    acc_on_batch = 0
    for k, (batch_input, batch_label) in enumerate(dev):
        pred, loss = apply(model, ctc_loss, batch_input, batch_label)
        dev_loss.append(loss.item())
        pred_words = greedy_decoding(pred)
        gold_list = [idx_to_class[word.item()] for word in batch_label]
        gold_set = set([(idx_to_class[word.item()], word_index) for word_index, word in enumerate(batch_label)])
        pred_set = set([(word, word_index) for word_index, word in enumerate(pred_words)])
        acc_on_batch += len(pred_set.intersection(gold_set)) / len(pred_set)
        total_cer = (np.array(list(map(lambda x: cer(x[0], x[1]), zip(pred_words, gold_list))))).mean()
    print_to_file(pred_words)
    return total_cer, acc_on_batch / len(batch_input)


def save_model(model_to_save):
    torch.save(model_to_save.state_dict(), PATH)


def train_model(model, train, dev):
    min_error_rate = 999
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    while len(loss_history) == 0 or loss_history[-1] < 0.1:
        for epoch in range(100):
            model.train()
            print("Epoch {}".format(epoch))
            for k, (batch_input, batch_label) in enumerate(train):
                optimizer.zero_grad()
                pred, loss = apply(model, ctc_loss, batch_input, batch_label)
                loss_history.append(loss.item())
                loss.backward()
                optimizer.step()

            char_error_rate, exact_acc = accuracy_on_dev(model, dev)
            print("acc is ")
            print(char_error_rate)
            error_rate.append(char_error_rate)
            save_to_file([loss_history, dev_loss, error_rate], "history.pickle")
            print("saved history to pickle")
            plot_loss(error_rate, loss_history, dev_loss, exact_acc)
            if len(error_rate) > 20 and error_rate[-1] < min_error_rate:
                min_error_rate = error_rate[-1]
                save_model(model)

    return model


train_model(speech_model, train_loader, valid_loader)
