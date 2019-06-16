from gcommand_loader import GCommandLoader
import torch
ctc_loss = torch.nn.CTCLoss()
dataset = GCommandLoader('./data/train/')
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)


for k, (input,label) in enumerate(test_loader):
    print(input.size(), len(label))
