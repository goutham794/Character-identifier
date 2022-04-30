from dialogue_loader import create_Sheldon_uterances_dataset, BucketBatchSampler, tokenizer, collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import cuda
import torch
from tqdm import tqdm
import numpy as np

from model import DistillBERT
import config





dataset = load_dataset('json', data_files='tbbt_dialogues.json', split="train")

dataset_dict = dataset.train_test_split(test_size=config.VALIDATION_SPLIT)

sheldon_utterances_dataset_train = dataset_dict['train'].map(create_Sheldon_uterances_dataset, remove_columns=["Character"])
sheldon_utterances_dataset_valid = dataset_dict['test'].map(create_Sheldon_uterances_dataset, remove_columns=["Character"])


bucket_batch_sampler_train = BucketBatchSampler(config.TRAIN_BATCH_SIZE, sheldon_utterances_dataset_train)
bucket_batch_sampler_valid = BucketBatchSampler(config.VALID_BATCH_SIZE, sheldon_utterances_dataset_valid)


train_dataloader = DataLoader(sheldon_utterances_dataset_train, batch_sampler = bucket_batch_sampler_train, shuffle=False, drop_last=False, collate_fn=collator)
valid_dataloader = DataLoader(sheldon_utterances_dataset_valid, batch_sampler = bucket_batch_sampler_valid, shuffle=False, drop_last=False, collate_fn=collator)


# for b in train_dataloader:
#     print(b)
#     break

# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'

model = DistillBERT()
model.to(device)


def loss_fn(outputs, targets):
    # print(outputs.shape[0])
    if targets.shape[0]==1:
        targets = targets.squeeze()
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)


def eval_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["encoded_utterances"]
            mask = d["mask"]
            targets = d["speaker_Sheldon"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        f1_score = metrics.f1_score(fin_targets, fin_outputs)
    return f1_score


def fit(num_epochs, model, loss_fn, opt, train_dl, valid_dl):
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for _,data in tqdm(enumerate(train_dl, 0)):
          ids = data['encoded_utterances'].to(device, dtype = torch.long)
          mask = data['mask'].to(device, dtype = torch.long)
          targets = data['speaker_Sheldon'].to(device, dtype = torch.float)
          outputs = model(ids, mask).squeeze()
          loss = loss_fn(outputs, targets)
          loss.backward()
          opt.step()
          opt.zero_grad()

        valid_acc = eval_fn(valid_dl, model)
        print('Epoch [{}/{}], Train Loss: {:.4f} and Validation acc {:.4f} and loss {:.4f}'.format(epoch+1, num_epochs, loss.item(),valid_acc, 1.1))


fit(1, model, loss_fn, optimizer, train_dataloader, valid_dataloader)