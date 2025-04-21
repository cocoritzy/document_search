import torch

class CBOW(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.embeddings = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.linear = torch.nn.Linear(in_features=emb, out_features=voc)

  def forward(self, inpt):
    emb = self.embeddings(inpt)
    emb = emb.mean(dim=1)
    out = self.linear(emb)
    return out
