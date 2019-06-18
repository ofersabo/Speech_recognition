from our_model import *
from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn

from our_model import get_seq_length

ctc_loss = torch.nn.CTCLoss(blank=26)
train_dataset = GCommandLoader('./data/train/')
valid_dataset = GCommandLoader('./data/valid/')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

c2i = "abcdefghijklmnopqrstuvwxyz"

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = our_model().to(device)


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
    pred = pred.view(rnn_seq_length, pred.size(0), number_of_classes)
    loss = ctc_loss(pred, targets, lengths, target_lengths)
    return pred, loss


criterion = nn.NLLLoss(size_average=False)


def greedy_decoding(pred):
    '''

    :param pred: batch of probability shape (101,100,27)
    :return: word
    '''
    list_of_words = []
    for seq in pred.view(pred.size(1), rnn_seq_length, number_of_classes):
        chars = torch.argmax(seq, dim=1).tolist()
        word = [chars[0]] + [c for index, c in enumerate(chars[1:], start=1) if c != chars[index - 1]]
        word = [c2i[i] for i in word if i != 26]
        word = "".join(word)
        # print(word)
        list_of_words.append(word)
    return list_of_words


def accuracy_on_dev(model, dev):
    model.eval()
    total_acc = 0
    for k, (batch_input, batch_label) in enumerate(dev):
        pred, loss = apply(model, ctc_loss, batch_input, batch_label)
        words = greedy_decoding(pred)
        gold_set = set([(idx_to_class[word.item()],word_index) for word_index,word in enumerate(batch_label)])
        pred_set = set([(word,word_index) for word_index,word in enumerate(words)])
        total_acc += len(pred_set.intersection(gold_set)) / len(pred_set)
    return total_acc / k


def train_model(model, train, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    ctc_loss = torch.nn.CTCLoss()
    for epoch in range(100):
        print("Epoch {}".format(epoch))
        for k, (batch_input, batch_label) in enumerate(train):
            model.train()
            optimizer.zero_grad()
            pred, loss = apply(model, ctc_loss, batch_input, batch_label)
            loss.backward()
            optimizer.step()
            print(loss)

        acc = accuracy_on_dev(model, dev)
        print("acc is ")
        print(acc)

    return model


train_model(model, train_loader, valid_loader)


