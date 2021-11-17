import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import constant
from utils.vocab import Vocab

SMALL = 1e-08
# 0: tokens, 1: pos, 2: mask_s, 3: labels
class ToyNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.K = args.K
        self.rnn_hidden = args.rnn_hidden
        self.max_sent_len = args.max_sent_len
        print("loading pretrained emb......")
        self.emb_matrix = np.load(args.dset_dir+'/'+args.dataset+'/embedding.npy')
        print("loading dataset vocab......")
        self.vocab = Vocab(args.dset_dir+'/'+args.dataset+'/vocab.pkl')

        # create embedding layers
        self.emb = nn.Embedding(self.vocab.size, args.emb_dim, padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), args.pos_dim) if args.pos_dim > 0 else None

        # initialize embedding with pretrained word embeddings
        self.init_embeddings()

        # dropout
        self.input_dropout = nn.Dropout(args.input_dropout)

        # GRU for P(Trc|S,Y')
        self.GRU_mean_rc = torch.nn.GRUCell(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.GRU_std_rc = torch.nn.GRUCell(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)

        # GRU for P(Tner|S,Y')
        self.GRU_mean_ner = torch.nn.GRUCell(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.GRU_std_ner = torch.nn.GRUCell(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        
        # define r
        self.r_mean_rc = nn.Linear(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.r_mean_rc2 = nn.Linear(self.rnn_hidden*2, self.rnn_hidden*2)
        self.r_std_rc = nn.Linear(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.r_std_rc2 = nn.Linear(self.rnn_hidden*2, self.rnn_hidden*2)
        self.r_mean_ner = nn.Linear(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.r_mean_ner2 = nn.Linear(self.rnn_hidden*2, self.rnn_hidden*2)
        self.r_std_ner = nn.Linear(len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID), self.rnn_hidden*2)
        self.r_std_ner2 = nn.Linear(self.rnn_hidden*2, self.rnn_hidden*2)

        # define encoder for the sharing representations S
        self.BiLSTM = LSTMRelationModel(args)

        # classifer1
        self.Lr1 = nn.Linear(4*self.rnn_hidden, 2*self.rnn_hidden)
        self.Cr1 = nn.Linear(2*self.rnn_hidden, len(constant.LABEL_TO_ID))
        self.Cg1 = nn.Linear(2*self.rnn_hidden, len(constant.BIO_TO_ID))
        
        # classifer2
        self.Lr2 = nn.Linear(4*self.rnn_hidden, 2*self.rnn_hidden)
        self.Cr2 = nn.Linear(2*self.rnn_hidden, len(constant.LABEL_TO_ID))
        self.Cg2 = nn.Linear(2*self.rnn_hidden, len(constant.BIO_TO_ID))
        
        # Fn 
        self.logsoft_fn1 = nn.LogSoftmax(dim=2)
        self.logsoft_fn2 = nn.LogSoftmax(dim=3)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
            
    def get_statistics_batch(self, embeds, task):
        if task == 'rc':
            Y = embeds[0].reshape(-1, len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID))
            H = embeds[1].reshape(-1, self.rnn_hidden*2)
            mean = self.GRU_mean_rc(Y, H) 
            std = self.GRU_std_rc(Y, H) 
            # reshape to bsz, sqlen, hidden_size
            mean = mean.reshape(embeds[1].size(0), embeds[1].size(1), -1)
            std = std.reshape(embeds[1].size(0), embeds[1].size(1), -1)
        elif task == 'ner':
            Y = embeds[0].reshape(-1, len(constant.BIO_TO_ID)+len(constant.LABEL_TO_ID))
            H = embeds[1].reshape(-1, self.rnn_hidden*2)
            mean = self.GRU_mean_ner(Y, H) 
            std = self.GRU_std_ner(Y, H) 
            # reshape to bsz, sqlen, hidden_size
            mean = mean.reshape(embeds[1].size(0), embeds[1].size(1), -1)
            std = std.reshape(embeds[1].size(0), embeds[1].size(1), -1)
        elif task == 'rc_r': # t|y'
            mean = self.r_mean_rc2(F.tanh(self.r_mean_rc(embeds)))
            std = self.r_std_rc2(F.tanh(self.r_std_rc(embeds)))
        elif task == 'ner_r': # t|y' 
            mean = self.r_mean_ner2(F.tanh(self.r_mean_ner(embeds)))
            std = self.r_std_ner2(F.tanh(self.r_std_ner(embeds)))
        cov = std * std + SMALL
        return mean, cov

    def get_sample_from_param_batch(self, mean, cov, sample_size):
        bsz, seqlen, tag_dim =  mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).cuda()
        z = z * torch.sqrt(cov).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)
        return z

    def kl_div(self, param1, param2, real_len, mask_kl):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        mean1, cov1 = param1
        mean2, cov2 = param2
        bsz, seqlen, tag_dim = mean1.shape
        var_len = tag_dim * real_len

        cov2_inv = 1 / cov2
        mean_diff = mean2 - mean1

        mean_diff = mean_diff.view(bsz, -1)
        cov1 = cov1.view(bsz, -1)
        cov2 = cov2.view(bsz, -1)
        cov2_inv = cov2_inv.view(bsz, -1)
        mask_kl = mask_kl.view(bsz, -1)

        temp = (mean_diff * cov2_inv*mask_kl).view(bsz, 1, -1)
        KL = 0.5 * (torch.sum(torch.log(cov2)*mask_kl ,dim=1) - torch.sum(torch.log(cov1)*mask_kl, dim=1) - var_len
                    + torch.sum(cov2_inv * cov1*mask_kl, dim=1) + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
        return KL
        
    # 0: tokens, 1: pos, 2: mask_s
    def forward(self, inputs, num_sample=1):

        tokens, pos, mask_s = inputs
        tokens_embs = self.emb(tokens)
        rnn_inputs = [tokens_embs]
        if self.args.pos_dim > 0:
            rnn_inputs += [self.pos_emb(pos)]
        rnn_inputs = torch.cat(rnn_inputs, dim=2)    
        lens = mask_s.sum(dim=1)
        rnn_inputs = self.input_dropout(rnn_inputs)
        H = self.BiLSTM((rnn_inputs, lens)) 

        # mask
        s_len = H.size(1)
        mask_NER = mask_s.unsqueeze(-1).repeat(1, 1, len(constant.BIO_TO_ID))
        mask_tmp = mask_s.unsqueeze(-1).repeat(1, 1, len(constant.LABEL_TO_ID)) 
        mask_tmp = mask_tmp.unsqueeze(1).repeat(1, s_len, 1, 1)
        mask_RC = torch.zeros_like(mask_tmp)
        real_len = mask_s.sum(dim=1).int()
        for i in range(mask_tmp.size(0)):
            mask_RC[i, :real_len[i], :real_len[i], :] = mask_tmp[i, :real_len[i], :real_len[i], :]
        Hg = H
        Hr = H
        
        # Cg get y1'
        logits_Cg = self.Cg1(Hg)
        prob_Cg = F.softmax(logits_Cg, dim=2)
        logits_Cg = self.logsoft_fn1(logits_Cg)
        logits_Cg1 = logits_Cg * mask_NER

        # Cr get y2'
        e1 = Hr.unsqueeze(2).repeat(1, 1, s_len, 1)
        e2 = Hr.unsqueeze(1).repeat(1, s_len, 1, 1)
        e12 = torch.cat([e1, e2], dim=3)
        e12 = F.relu(self.Lr1(e12), inplace=True)
        del e1
        del e2
        prob_Cr = torch.sigmoid(self.Cr1(e12))
        del e12
        prob_Cr = prob_Cr * mask_RC
        logits_Cr1 = prob_Cr
        prob_Cr = torch.where(mask_RC==0, torch.zeros_like(prob_Cr)-10e10, prob_Cr)
        prob_Cr = prob_Cr.max(dim=2)[0]
         
        # P(Trc|S,Y') 
        Y = torch.cat([prob_Cr, prob_Cg], dim=2)
        mean_rc, cov_rc = self.get_statistics_batch((Y, H), task='rc')
        encoding_rc = self.get_sample_from_param_batch(mean_rc, cov_rc, num_sample)
        # P(Tner|S,Y')
        mean_ner, cov_ner = self.get_statistics_batch((Y, H), task='ner')
        encoding_ner = self.get_sample_from_param_batch(mean_ner, cov_ner, num_sample)
        # P(Trc|Y')
        mean_r_rc, cov_r_rc = self.get_statistics_batch(torch.tensor(Y).cuda(), task='rc_r')
        # P(Tner|Y')
        mean_r_ner, cov_r_ner = self.get_statistics_batch(torch.tensor(Y).cuda(), task='ner_r')
        
        # repeat mask
        mask_NER = mask_NER.unsqueeze(1).repeat(1, num_sample, 1, 1)
        mask_RC = mask_RC.unsqueeze(1).repeat(1, num_sample, 1, 1, 1)
        
        Hg = encoding_ner
        Hr = encoding_rc
        
        # use compressed Hg and Hr for classification 
        logits_Cg = self.Cg2(Hg)
        logits_Cg = self.logsoft_fn2(logits_Cg)
        logits_Cg2 = logits_Cg * mask_NER

        e1 = Hr.unsqueeze(3).repeat(1, 1, 1, s_len, 1)
        e2 = Hr.unsqueeze(2).repeat(1, 1, s_len, 1, 1)
        e12 = torch.cat([e1, e2], dim=4)
        e12 = F.relu(self.Lr2(e12), inplace=True)
        del e1
        del e2
        logits_Cr = torch.sigmoid(self.Cr2(e12))
        del e12
        logits_Cr2 = logits_Cr *  mask_RC
        
        # caculate KL divergence for encoding_rc
        seqlen, bsz = s_len, mask_s.size(0) 
        mask_kl = mask_s.unsqueeze(-1).repeat(1, 1, self.K)
        mean, cov = mean_rc, cov_rc
        mean_r, cov_r = mean_r_rc, cov_r_rc
        kl_div_rc = self.kl_div((mean, cov), (mean_r, cov_r), real_len, mask_kl)
        
        # caculate KL divergence for encoding_ner
        mean, cov = mean_ner, cov_ner
        mean_r, cov_r = mean_r_ner, cov_r_ner
        kl_div_ner = self.kl_div((mean, cov), (mean_r, cov_r), real_len, mask_kl)
            
        return self.args.beta1*kl_div_rc.mean()+self.args.beta2*kl_div_ner.mean(), logits_Cg1.view(-1, len(constant.BIO_TO_ID)), logits_Cr1.view(-1, len(constant.LABEL_TO_ID)), logits_Cg2.view(-1, len(constant.BIO_TO_ID)), logits_Cr2.view(-1, len(constant.LABEL_TO_ID))


# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.emb_dim + args.pos_dim
        self.rnn = nn.LSTM(self.in_dim, self.args.rnn_hidden, 1, batch_first=True, \
                               dropout=0, bidirectional=True)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, 1, True)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        # unpack inputs
        inputs, lens = inputs[0], inputs[1]
        return self.encode_with_rnn(inputs, lens, inputs.size()[0])

# Initialize zero state
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0




