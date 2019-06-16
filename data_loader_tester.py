from RNN import RNN
from gcommand_loader import GCommandLoader
import torch

ctc_loss = torch.nn.CTCLoss()
train_dataset = GCommandLoader('./data/train/')
valid_dataset = GCommandLoader('./data/valid/')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

c2i = "abcdefghijklmnopqrstuvwxyz"

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=100, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)



n_hidden = 128
rnn = RNN(161, n_hidden, 27)

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def train(line_tensor,label):
    ctc_loss = torch.nn.CTCLoss()
    for batch_index in range(line_tensor.size()[0]):
        rnn.zero_grad()
        hidden = rnn.initHidden()
        flat_input = line_tensor[batch_index].reshape((101,161))
        proba_matrix = []
        for time_frame in flat_input:
            output, hidden = rnn(time_frame, hidden)
            proba_matrix.append(output)
        stacked_tensor = torch.stack(proba_matrix)
        stacked_tensor = stacked_tensor.reshape((stacked_tensor.size()[0],1,-1))

        # Initialize random batch of targets (0 = blank, 1:C+1 = classes)
        word = idx_to_class[label[0].item()]
        chars = []
        for c in word:
            chars.append(c2i.index(c))
        target = torch.tensor(chars) #TODO  add padding when using batch

        input_lengths = torch.full(size=(1,), fill_value=101, dtype=torch.long)
        target_lengths = torch.full(size=(1,), fill_value=len(chars),dtype=torch.long)

        loss = ctc_loss(stacked_tensor, target, input_lengths, target_lengths)
        print(loss.item())
        loss.backward()



def predict(line_tensor,label):
    for batch_index in range(line_tensor.size()[0]):
        rnn.zero_grad()
        hidden = rnn.initHidden()
        flat_input = line_tensor[batch_index].reshape((101,161))
        chars = []
        for time_frame in flat_input:
            output, hidden = rnn(time_frame, hidden)
            chars.append(output.argmax().item())
        word = [chars[0]] + [c for index, c in enumerate(chars[1:],start=1) if c!=chars[index-1] ]
        word = [c2i[i] for i in word if i != 27]
        word = "".join(word)
        print(word)
        # Initialize random batch of targets (0 = blank, 1:C+1 = classes)


# for k, (input, label) in enumerate(train_loader):
#     train(input,label)
#     if k>1:
#         break


rnn.eval()
for k, (input, label) in enumerate(valid_dataset):
    predict(input,label)
    # print(input.size(), len(label))
