from model import DistillBERT
from torch.utils.data import DataLoader
from torch import cuda
from tqdm import tqdm
import torch
from sklearn import metrics
from datasets import load_dataset
import numpy as np

import config
from dialogue_loader import create_Sheldon_uterances_dataset, BucketBatchSampler, collator


device = 'cuda' if cuda.is_available() else 'cpu'


model = DistillBERT()
model.to(device)


checkpoint = torch.load(config.MODEL_SAVE_PATH )
model.load_state_dict(checkpoint['model_state_dict'])

def loss_fn(outputs, targets):
    # print(outputs.shape[0])
    if targets.shape[0]==1:
        targets = targets.squeeze()
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    

def eval_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_loss = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["encoded_utterances"]
            mask = d["mask"]
            targets = d["speaker_Sheldon"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs.squeeze(), targets)
            total_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        f1_score = metrics.f1_score(fin_targets, fin_outputs)
    return f1_score, total_loss

dataset = load_dataset('json', data_files='tbbt_dialogues.json', split="train")

dataset_dict = dataset.train_test_split(test_size=config.VALIDATION_SPLIT)

sheldon_utterances_dataset_train = dataset_dict['train'].map(create_Sheldon_uterances_dataset,)
sheldon_utterances_dataset_valid = dataset_dict['test'].map(create_Sheldon_uterances_dataset, )#remove_columns=["Character"])

train_dataloader = DataLoader(sheldon_utterances_dataset_train,  batch_size =config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator)
valid_dataloader = DataLoader(sheldon_utterances_dataset_valid,  batch_size =config.VALID_BATCH_SIZE, shuffle=True, collate_fn=collator)


f1_score, loss = eval_fn(train_dataloader, model)
valid_f1_score, valid_loss = eval_fn(valid_dataloader, model)

print(f"F1 score {f1_score} and loss {loss}")
print(f"valid F1 score {valid_f1_score} and valid loss {valid_loss}")