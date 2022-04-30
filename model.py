import torch
from transformers import DistilBertModel



class DistillBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output