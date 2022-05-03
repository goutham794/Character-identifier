from dialogue_loader import create_Sheldon_uterances_dataset, BucketBatchSampler, collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import cuda
import torch

from model import DistillBERT
import config
import engine
from utils import SaveBestModel

save_best_model = SaveBestModel()


###############
# LOAD THE DATA
###############

dataset = load_dataset('json', data_files='tbbt_dialogues.json', split="train")

dataset_dict = dataset.train_test_split(test_size=config.VALIDATION_SPLIT)

sheldon_utterances_dataset_train = dataset_dict['train'].map(create_Sheldon_uterances_dataset,) #remove_columns=["Character"])
sheldon_utterances_dataset_valid = dataset_dict['test'].map(create_Sheldon_uterances_dataset, )#remove_columns=["Character"])



bucket_batch_sampler_train = BucketBatchSampler(config.TRAIN_BATCH_SIZE, sheldon_utterances_dataset_train)
bucket_batch_sampler_valid = BucketBatchSampler(config.VALID_BATCH_SIZE, sheldon_utterances_dataset_valid)


train_dataloader = DataLoader(sheldon_utterances_dataset_train, batch_sampler = bucket_batch_sampler_train, shuffle=False, drop_last=False, collate_fn=collator)
valid_dataloader = DataLoader(sheldon_utterances_dataset_valid, batch_sampler = bucket_batch_sampler_valid, shuffle=False, drop_last=False, collate_fn=collator)
# train_dataloader = DataLoader(sheldon_utterances_dataset_train,  batch_size =config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator)
# valid_dataloader = DataLoader(sheldon_utterances_dataset_valid,  batch_size =config.VALID_BATCH_SIZE, shuffle=True, collate_fn=collator)


# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'

###############
# GET THE MODEL
###############

model = DistillBERT()
model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(config.NUM_EPOCHS):
    engine.fit(model, optimizer, train_dataloader, device)
    train_f1, train_loss = engine.eval_fn(train_dataloader, model, device)
    valid_f1, valid_loss = engine.eval_fn(valid_dataloader, model, device)
    print('Epoch [{}/{}], Train Loss: {:.4f} Train f1 {:.4f} and Validation loss {:.4f} and f1 {:.4f}'.format(epoch+1, config.NUM_EPOCHS, train_loss,train_f1, valid_loss, valid_f1))
    save_best_model(valid_loss, epoch, model, optimizer)