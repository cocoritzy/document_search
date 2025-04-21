import torch

class CBOW(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)

  def forward(self, inpt):
    emb = self.emb(inpt)
    emb = emb.mean(dim=1)
    out = self.ffw(emb)
    return out
