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

        # define r distribution
        self.r_var = self.K * self.max_sent_len
        self.r_mean_rc = nn.Parameter(torch.randn(self.max_sent_len, self.K))
        self.r_mean_ner = nn.Parameter(torch.randn(self.max_sent_len, self.K))
        self.r_std_rc = nn.Parameter(torch.randn(self.max_sent_len, self.K))
        self.r_std_ner = nn.Parameter(torch.randn(self.max_sent_len, self.K))

        # define encoder
        self.BiLSTM = LSTMRelationModel(args)
        self.hidden2mean_rc = nn.Linear(self.rnn_hidden*2, self.K)
        self.hidden2mean_ner = nn.Linear(self.rnn_hidden*2, self.K)
        self.hidden2std_rc = nn.Linear(self.rnn_hidden*2, self.K)
        self.hidden2std_ner = nn.Linear(self.rnn_hidden*2, self.K)

        # decoder
        self.lr_rc = nn.Linear(args.K*2, args.K)
        self.rc_cla = nn.Linear(args.K, len(constant.LABEL_TO_ID))
        self.ner_cla = nn.Linear(args.K, len(constant.BIO_TO_ID))
        self.logsoft_fn = nn.LogSoftmax(dim=3)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

    def get_statistics_batch(self, embeds, task):
        if task == 'rc':
            mean = self.hidden2mean_rc(embeds) # bsz, seqlen, dim
            std = self.hidden2std_rc(embeds) # bsz, seqlen, dim
        if task == 'ner':
            mean = self.hidden2mean_ner(embeds) # bsz, seqlen, dim
            std = self.hidden2std_ner(embeds) # bsz, seqlen, dim
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
        # construct input feature X
        tokens, pos, mask_s = inputs
        tokens_emb = self.emb(tokens)
        tokens_emb = [tokens_emb]
        if self.args.pos_dim > 0:
            tokens_emb += [self.pos_emb(pos)]
        tokens_emb = torch.cat(tokens_emb, dim=2)
        lens = mask_s.sum(dim=1)
        tokens_emb = self.input_dropout(tokens_emb)

        # forward into BiLSTM
        temp = self.BiLSTM((tokens_emb, lens)) # bsz, len, K
        # encode t
        mean_rc, cov_rc = self.get_statistics_batch(temp, 'rc')
        mean_ner, cov_ner = self.get_statistics_batch(temp, 'ner')
        encoding_rc = self.get_sample_from_param_batch(mean_rc, cov_rc, num_sample)
        encoding_ner = self.get_sample_from_param_batch(mean_ner, cov_ner, num_sample)
        
        # mask for output
        s_len = encoding_rc.size(2)
        ner_mask = mask_s.unsqueeze(-1).expand(-1, -1, len(constant.BIO_TO_ID))
        tmp_mask = mask_s.unsqueeze(-1).expand(-1, -1, len(constant.LABEL_TO_ID))
        tmp_mask = tmp_mask.unsqueeze(1).expand(-1, s_len, -1, -1)
        rc_mask = torch.zeros_like(tmp_mask)
        real_len = mask_s.sum(dim=1).int()
        for i in range(tmp_mask.size(0)):
            rc_mask[i, :real_len[i], :real_len[i], :] = tmp_mask[i, :real_len[i], :real_len[i], :]
        ner_mask = ner_mask.unsqueeze(1).expand(-1, num_sample, -1, -1)
        rc_mask = rc_mask.unsqueeze(1).expand(-1, num_sample, -1, -1, -1)

        # ner prediction
        ner_logit = self.ner_cla(encoding_ner)
        ner_logit = self.logsoft_fn(ner_logit)
        ner_logit = ner_logit * ner_mask

        # rc prediction
        encoding_e1 = encoding_rc.unsqueeze(3).expand(-1, -1, -1, s_len, -1) # bsz, sample_num, len, len, K
        encoding_e2 = encoding_rc.unsqueeze(2).expand(-1, -1, s_len, -1, -1)
        encoding_e = torch.cat([encoding_e1, encoding_e2], dim=4)
        encoding_e = F.relu(self.lr_rc(encoding_e), inplace=True)
        del encoding_e1
        del encoding_e2
        rc_logit = torch.sigmoid(self.rc_cla(encoding_e))
        rc_logit = rc_logit * rc_mask
        del encoding_e

        # caculate KL divergence for the rc encoding
        seqlen, bsz = s_len, mask_s.size(0) 
        mask_kl = mask_s.unsqueeze(-1).repeat(1, 1, self.K)
        mean_r = self.r_mean_rc[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        std_r = self.r_std_rc[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        cov_r = std_r * std_r + SMALL
        mean, cov = mean_rc, cov_rc
        mean_r, cov_r = mean_r, cov_r
        kl_div_rc = self.kl_div((mean, cov), (mean_r, cov_r), real_len, mask_kl)

        # caculate KL divergence for the ner encoding
        mean_r = self.r_mean_ner[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        std_r = self.r_std_ner[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
        cov_r = std_r * std_r + SMALL
        mean, cov = mean_ner, cov_ner
        mean_r, cov_r = mean_r, cov_r
        kl_div_ner = self.kl_div((mean, cov), (mean_r, cov_r), real_len, mask_kl) 

        return self.args.beta1*kl_div_rc.mean()+self.args.beta2*kl_div_ner.mean(), torch.zeros(1).cuda()[0], ner_logit.view(-1, len(constant.BIO_TO_ID)), rc_logit.view(-1, len(constant.LABEL_TO_ID))

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