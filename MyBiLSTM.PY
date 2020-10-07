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
#creat model，nn.module是所有网络的父类
class BiLSTM_CRF(nn.Module):

    #实例化的方法
    def __init__(self,vocab_size,tag_to_ix,embedding_dim,hidden_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        #标签控件
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        #这里使用pytorch的embeddin
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        #nn.LSTM(当前时刻的输入向量长度，隐藏层维数，层数，是否双向) 若双向，单向隐藏层拼接故为总向量//2
        #实例化一个LSTM网络
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        #增加一个线性层从LSTM的输出(隐藏层维数)映射到CRF的tag维数
        self.hidden2tag = nn.Linear(hidden.dim, self.tagset_size)

        #转移矩阵要一起训练，所以定义一个可求梯度的矩阵，parameter类继承自tensor，默认可求梯度
        self.transitions = nn.parameter(
            torch.randn(self.tagset_size,self.tagset_size )
        )

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        #初始化隐藏层h0，c0
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn()

        )

    #Negative log-likehood 反对数似然作为损失函数
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        full_path_score = self._full_path_score(feats)
        gold_score = self._realtag_score(feats,tags)
        return full_path_score - gold_score

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        #LSTM的输入(imput_tensor,init_hidden_tensor)
        #batch_first默认值为False,则input_tensor（sentence_len, batch, input_size）
        #LSTM的输出(output_tensor,(h_n,c_n))
        # output_tensor（seq_len, batch, num_directions * hidden_size）
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)



    #计算所有路径的对数 softmanx分数，就是用动态规划计算的那一项，输入全部的序列LSTM计算后的矩阵 标签维数X序列长度
    #
    def _full_path_score(self, feats):
        init_prelist = torch.full((1, self.tagset_size), -10000.)
        init_prelist[0][tag_to_ix[START_TAG]] = 0


        for feat in feats:
