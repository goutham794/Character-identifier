from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import torch

from collections import OrderedDict
from random import shuffle

from transformers import DistilBertTokenizer

# Loading tokenizer.
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")




def create_Sheldon_uterances_dataset(row):
    """
    Add new feature `speaker_Sheldon` 1 - Sheldon is the speaker, 0 otherwise.
    """
    character = row['Character']
    speaker_Sheldon = 0
    if character == "Sheldon":
        speaker_Sheldon = 1
    return {"speaker_Sheldon": speaker_Sheldon}




class BucketBatchSampler(Sampler):

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        # For each data item we store the index and the length of utterance + context.
        self.ind_n_len = []
        for i in range(len(data)):
            self.ind_n_len.append((i, len(data[i]['Line'])))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)


    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # batch_map is a dictionary with length as key and the indices as value.
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list
    
    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        # shuffle all the batches so they aren't ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
    



def collator(batch):
    # print(f"batch_size: {len(batch)}")
    utterances_batch = []
    encoded_utterance_batch = []
    speaker_Sheldon_batch = []
    utterance_mask_batch = []
    for example in batch:
        utterances_batch.append(example['Line'])
        speaker_Sheldon_batch.append(example['speaker_Sheldon'])

   
    for utterance in utterances_batch:
        encoded_utterance_single = torch.tensor(tokenizer(utterance)['input_ids'], dtype=torch.long)
        encoded_utterance_batch.append(encoded_utterance_single)
        utterance_mask_batch.append(torch.tensor([1]*len(encoded_utterance_single)))
    
    encoded_utterance_batch = pad_sequence(encoded_utterance_batch, batch_first=True)
    utterance_mask_batch = pad_sequence(utterance_mask_batch, batch_first=True)

    
    return {
        "encoded_utterances" : encoded_utterance_batch,
        "speaker_Sheldon" : torch.tensor(speaker_Sheldon_batch, dtype=torch.float),
        "mask" : utterance_mask_batch
        # "Lines" : utterances_batch
    }