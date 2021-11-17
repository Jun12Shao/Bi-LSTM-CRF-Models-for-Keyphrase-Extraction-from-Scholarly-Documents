# -*- coding:utf-8 -*-
import os
import torch.nn as nn
import torch.autograd as autograd
import torch
import pickle


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):       ## the size of vec is (tag_size)
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



def log_sum_exp2(vec):       ## the size of vec is (bt_size, tag_size)
    # max_score = vec[0, argmax(vec)]
    # max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    bt_size=vec.shape[0]

    max_score,_=torch.max(vec,1)
    max_score_broadcast = max_score.view(bt_size, -1).expand(-1, vec.shape[1])
    max_score=max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),dim=1))

    return max_score

class BiLSTM_CRF(nn.Module):

    def __init__(self,args, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.args=args
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.load_check_point=args.load_check_point
        self.set_and_check_load_epoch()

        self.init_embedding()
        ## create and initiate Bi_LSTM_CRF
        self.create_network()



    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, self.args.batch_size, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, self.args.batch_size, self.hidden_dim // 2)))

    def init_embedding(self):
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.args.pretrained_embedding:
            emb_path=os.path.join(self.args.root,self.args.pretrained_embedding)
            with open(emb_path, 'rb') as f:
                embedding = pickle.load(f, encoding='latin1')
                f.close()
            self.word_embeds.weight.data.copy_(torch.from_numpy(embedding))
            print("Initialized with glove embedding")
        else:
            print("Random initialized embedding")

    def create_network(self):
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,  ## the number of features in hidden state h.
                            num_layers=self.args.layers, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.args.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[self.args.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[self.args.STOP_TAG]] = -10000
        self.hidden = self.init_hidden()



    def _forward_alg(self, feats, mask):        ## feats (length,batch_size,tag_size)
        # Do the forward algorithm to compute the partition function
        ##remove the start, padding and end notation at first.  count is the number of effective words in the senquence.
        bt_size = feats.shape[1]
        length=feats.shape[0]

        init_alphas = torch.full((bt_size, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:,self.tag_to_ix[self.args.START_TAG]] = 0.    ## The begining should be "start" tag

        # Wrap in a variable so that we will get automatic backprop
        forward_var=torch.full((length+1, bt_size, self.tagset_size), 0.)
        forward_var[0] = init_alphas

        # Iterate through the sentence
        for step, feat in enumerate(feats[1:-1]): ## feats don't contain start and end.
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:,next_tag].view(bt_size, -1).expand(-1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).expand(bt_size,-1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var[step] + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp2(next_tag_var))    ##注意调整
            forward_var[step+1] = torch.stack(alphas_t,dim=1).view(bt_size, -1)

        # terminal_var = forward_var + self.transitions[self.tag_to_ix[self.args.STOP_TAG]].view(1,-1).expand(bt_size,-1)
        terminal_var=[[]]*bt_size
        for i in range(bt_size):
            idx=mask[i]
            terminal_var[i]=forward_var[idx,i]+self.transitions[self.tag_to_ix[self.args.STOP_TAG]]

        alpha=torch.zeros(1)
        for vec in terminal_var:
            alpha+=log_sum_exp(vec.view(1,-1))
        alpha/=bt_size

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        embeds=embeds.transpose(0,1)        ##(length,batch_size,embed_dim)
        # embeds = self.word_embeds(sentence).view(sentence.shape[1], 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)   ##(length,batch_size,hidden_dim)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)           ##(length,batch_size,tag_size)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):   ## feats:(length,bt_size,tag_size)  tags:(bt_size,length)
        # Gives the score of a provided tag sequence
        bt_size=feats.shape[1]
        length=feats.shape[0]
        feats=feats[1:] ##remove start node for all sequences
        # tags=tags[1:]
        score = torch.zeros((length+1,bt_size))
        terminal_score=torch.zeros(1)

        for i, feat in enumerate(feats):
            for j in range(bt_size):
                score[i+1,j] = score[i,j] + self.transitions[tags[j, i+1], tags[j, i]] + feat[j, tags[j,i+1]]


        for i in range(bt_size):
            idx=mask[i]
            terminal_score += score[idx,i] + self.transitions[self.tag_to_ix[self.args.STOP_TAG], tags[i,idx]]

        return terminal_score/bt_size


    def neg_log_likelihood(self, sentence, tags, mask):
        feats = self._get_lstm_features(sentence)     ##(length,batch_size,tag_size)
        forward_score = self._forward_alg(feats,mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return forward_score - gold_score


    def _viterbi_decode(self, feats):  ##feats:(length,batch_size,lag_size)   mask:(batch_size)
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.args.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.args.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.args.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def predict_vb(self, sentence,mask):         ##(batch_size, length), (batch_size)
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)     ##(length,batch_size,tag_size)
        bt_size=len(mask)

        # Find the best path, given the features.
        scores=0
        tag_seqs=[]
        for i in range(bt_size):
            score, tag_seq = self._viterbi_decode(lstm_feats[1:mask[i]+1,i,:])
            scores+=score
            tag_seqs.append(tag_seq)

        return scores/bt_size, tag_seqs



    def set_and_check_load_epoch(self):
        if os.path.exists(self.load_check_point):
            if self.args.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(self.load_check_point):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self.args.load_epoch = load_epoch
                self.load_model_path=os.path.join(self.load_check_point,'net_epoch_%s_id.pth' % (load_epoch))
                self.load_opt_path=os.path.join(self.load_check_point,'opt_epoch_%s_id.pth' % (load_epoch))
            else:
                 found = False
                 for file in os.listdir(self.load_check_point):
                     if file.startswith("net_epoch_"):
                         found = int(file.split('_')[2]) == self.args.load_epoch
                         if found:  break
                 assert found, 'Model for epoch %i not found' % self.args.load_epoch
        else:
            assert self.args.load_epoch < 1, 'Model for epoch %i not found' % self.args.load_epoch
            self.args.load_epoch = 0
