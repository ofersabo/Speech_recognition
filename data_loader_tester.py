from our_model import *
import torch.nn as nn
from our_model import get_seq_length

from utils import *

ctc_loss = torch.nn.CTCLoss(blank=26)
c2i = "abcdefghijklmnopqrstuvwxyz"

model = our_model().to(device)
loss_history = []
dev_loss = []
total_acc = []


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
    pred = pred.permute(1,0,2)
    loss = ctc_loss(pred, targets, lengths, target_lengths)
    return pred, loss


criterion = nn.NLLLoss(size_average=False)


def greedy_decoding(pred):
    '''

    :param pred: batch of probability shape (101,100,27)
    :return: word
    '''
    list_of_words = []
    pred = pred.permute(1,0,2)
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
        words = greedy_decoding(pred)
        gold_set = set([(idx_to_class[word.item()], word_index) for word_index, word in enumerate(batch_label)])
        pred_set = set([(word, word_index) for word_index, word in enumerate(words)])
        acc_on_batch += len(pred_set.intersection(gold_set)) / len(pred_set)
    return acc_on_batch / k


def train_model(model, train, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    while len(loss_history) == 0 or loss_history[-1] < 0.1:
        for epoch in range(100):
            print("Epoch {}".format(epoch))
            for k, (batch_input, batch_label) in enumerate(train):
                model.train()
                optimizer.zero_grad()
                pred, loss = apply(model, ctc_loss, batch_input, batch_label)
                loss_history.append(loss.item())
                loss.backward()
                optimizer.step()

            acc = accuracy_on_dev(model, dev)
            print("acc is ")
            print(acc)
            total_acc.append(acc)
            save_to_file([loss_history, dev_loss, total_acc], "history.pickle")
            print("saved history to pickle")
            plot_loss(total_acc, loss_history, dev_loss)
    return model


train_model(model, train_loader, valid_loader)
