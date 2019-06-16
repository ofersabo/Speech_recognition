from RNN import RNN
from gcommand_loader import GCommandLoader
import torch

ctc_loss = torch.nn.CTCLoss()
dataset = GCommandLoader('./data/train/')
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

c2i = "abcdefghijklmnopqrstuvwxyz"

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=None,
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


    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


for k, (input, label) in enumerate(train_loader):
    train(input,label)
    # print(input.size(), len(label))
