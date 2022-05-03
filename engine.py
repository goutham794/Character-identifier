import torch
from tqdm.auto import tqdm
from sklearn import metrics
import numpy as np



def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))



def fit(model, opt, train_dl, device):
    model.train()
    for _,data in tqdm(enumerate(train_dl, 0)):
        ids = data['encoded_utterances'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['speaker_Sheldon'].to(device, dtype = torch.float)
        outputs = model(ids, mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        opt.step()
        opt.zero_grad()



def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_loss = 0
    with torch.no_grad():
        for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["encoded_utterances"]
            mask = d["mask"]
            targets = d["speaker_Sheldon"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        f1_score = metrics.f1_score(fin_targets, fin_outputs)
    return f1_score, total_loss