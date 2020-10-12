import torch
import torch.nn as nn
import torch.optim as optim
import ProcessDATA
import os
from tag2num import tag_to_ix
from num2tag import ix_to_tag
from word_dic import word_to_ix

START_TAG = "<START>"
STOP_TAG = "<STOP>"
# torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix:
            idxs.append(0)
        else:
            idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.)
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list=[]
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            t_r1_k = torch.unsqueeze(feats[feat_index],0).transpose(0,1)
            aa = gamar_r_l + t_r1_k + self.transitions
            forward_var_list.append(torch.logsumexp(aa,dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var,0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t,bptrs_t = torch.max(next_tag_var,dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t,0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__== '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 300#300
    HIDDEN_DIM = 256#256
    train_set_times = 3
    

    #处理训练数据
    get_training_data = ProcessDATA.get_train_data()
    
    train_set_num = [
        [0,len(get_training_data)//2],
        [len(get_training_data)//4,len(get_training_data)//4*3],
        [len(get_training_data)//2,len(get_training_data)]
    ]
    
    print('traing data',len(get_training_data))
    #准备字典 字->编号
    for sentence, tags in get_training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    # Make up some training data

    for num in train_set_num:
        print('train_set',num)
        training_data = get_training_data[num[0]:num[1]]

        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Check predictions before training
        with torch.no_grad():
            precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
            print(model(precheck_sent))
            print(precheck_tags)

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(
                3):  # again, normally you would NOT do 300 epochs, it is toy data
            print('epoch',epoch)
            for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

                # Step 3. Run our forward pass.
                loss = model.neg_log_likelihood(sentence_in, targets)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

    # Check predictions after training
    with torch.no_grad():

        fildir = os.path.join(os.getcwd(),'data','chusai_xuanshou')
        filenames = os.listdir(fildir)

        #print(filenames_txt)
        for filename in filenames:
            name = os.path.splitext(filename)[0]
            with open(os.path.join(fildir,'{}.txt'.format(name)),'r',encoding='utf-8') as f:
                txtdata = f.read()
                txtdata = txtdata.replace('\u3000',' ')
                print('prediction file ',filename)
            
            sentence_loca = []
            test_data = []
            test_data,sentence_loca = ProcessDATA.split_txt(txtdata)
            #print(test_data)
            #对每一个句子循环
            cont = 0
            for sent_num in range(len(test_data)):
                prediction_sent = prepare_sequence(test_data[sent_num], word_to_ix)
                ixtags = model(prediction_sent)[1]
                print(ixtags)
                with open(os.path.join(fildir,'{}.ann'.format(name)),'a',encoding='utf-8') as f:
                    for i in range(len(ixtags)):
                        if not ixtags[i] == 41:
                            if i == 0:#判定是否开头
                                cont = cont + 1
                                s = i
                                lenth = 1
                                if ixtags[i+1] == 41:#如果是开头的单字则输出
                                    f.write('T'+str(cont)+'\t'+ix_to_tag[ixtags[i]]+' '
                                            +str(sentence_loca[sent_num][0]+s)+' '
                                            +str(sentence_loca[sent_num][0]+s+lenth)+'\t'
                                            +test_data[sent_num][s:s+lenth]+'\n')

                            elif i== len(ixtags)-1:#判定是否结尾
                                if ixtags[i-1] == 41:#如果是结尾的单字需要赋值
                                    cont = cont + 1
                                    s = i
                                    lenth =1
                                else:
                                    lenth = lenth +1
                                
                                f.write('T'+str(cont)+'\t'+ix_to_tag[ixtags[i]]+' '
                                        +str(sentence_loca[sent_num][0]+s)+' '
                                        +str(sentence_loca[sent_num][0]+s+lenth)+'\t'
                                        +test_data[sent_num][s:s+lenth]+'\n')

                            else:#中间项
                                if ixtags[i-1] == 41:
                                    cont = cont + 1
                                    s = i
                                    lenth = 1
                                else:
                                    lenth = lenth +1

                                if ixtags[i+1] == 41:
                                    f.write('T'+str(cont)+'\t'+ix_to_tag[ixtags[i]]+' '
                                            +str(sentence_loca[sent_num][0]+s)+' '
                                            +str(sentence_loca[sent_num][0]+s+lenth)+'\t'
                                            +test_data[sent_num][s:s+lenth]+'\n')
        # We got it!