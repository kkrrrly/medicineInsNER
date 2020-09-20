import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

net = nn.LSTMCell

EMBEDDING_DIM = 39

trainin_data = [(
    "the wall street journal reported today that apple corporation made money".split(), #['the','wall','street','journal',...]
    "B I I I O O O B I O O".split() #['B','I','I',...]
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]
####################
#creat model
class BiLSTM_CRF(nn.Module):

    def __init__(self,embedding_dim,hidden_dim)
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim

    


model 
optimizer