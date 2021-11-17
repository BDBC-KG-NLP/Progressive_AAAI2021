import numpy as np
import torch
import argparse
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from loader import Dataloader
from utils.scorer import sta
from utils.vocab import Vocab
from model import ToyNet
from pathlib import Path
import os
from utils import constant

class Solver(object):

    def __init__(self, args):
        self.args = args

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.K = args.K
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0
        self.log_file = args.log_file

        # Network & Optimizer
        self.toynet = ToyNet(args).cuda()
        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr)
        
        self.ckpt_dir = Path(args.ckpt_dir)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # loss function
        self.ner_lossfn = nn.NLLLoss(reduction='sum')
        self.rc_lossfn = nn.BCELoss(reduction='sum')

        # History
        self.history = dict()
        # class loss
        self.history['ner_train_loss1'] = []
        self.history['rc_train_loss1'] = []
        self.history['ner_test_loss1'] = []
        self.history['rc_test_loss1'] = []
        self.history['ner_train_loss2'] = []
        self.history['rc_train_loss2'] = []
        self.history['ner_test_loss2'] = []
        self.history['rc_test_loss2'] = []
        self.history['precision_test'] = []
        self.history['recall_test'] = []
        self.history['F1_test'] = []
        # info loss
        self.history['info_train_loss'] = []
        self.history['info_test_loss'] = []

        # Dataset
        vocab = Vocab(args.dset_dir+'/'+args.dataset+'/vocab.pkl')
        self.data_loader = dict()
        self.data_loader['train'] = Dataloader(args.dset_dir+'/'+args.dataset+'/train.json', args.batch_size, vars(args), vocab)
        self.data_loader['test'] = Dataloader(args.dset_dir+'/'+args.dataset+'/test.json', args.batch_size, vars(args), vocab, evaluation=True)

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.toynet.train()
        elif mode == 'eval' :
            self.toynet.eval()
        else : raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        for e in range(self.epoch):
            self.global_epoch += 1
            ner_train_loss1, rc_train_loss1, ner_train_loss2, rc_train_loss2, info_train_loss = 0., 0., 0., 0., 0.
            local_iter = 0
            for inputs, ner_labels, rc_labels in self.data_loader['train']:
                self.global_iter += 1
                local_iter += 1

                inputs = [Variable(i).cuda() for i in inputs]
                mask_s = inputs[2]
                ner_labels = Variable(ner_labels).cuda()
                rc_labels = Variable(rc_labels).cuda() 
                info_train_loss_, ner_logit1, rc_logit1, ner_logit2, rc_logit2 = self.toynet(inputs, self.args.num_avg)

                # loss
                ner_train_loss_1 = self.ner_lossfn(ner_logit1, ner_labels.view(-1)) / ner_labels.size(0) 
                rc_train_loss_1 = self.rc_lossfn(rc_logit1, rc_labels.view(-1, len(constant.LABEL_TO_ID))) / rc_labels.size(0)
                ner_train_loss_2 = self.ner_lossfn(ner_logit2, ner_labels.unsqueeze(1).repeat(1,self.args.num_avg,1).view(-1)) / (ner_labels.size(0)*self.args.num_avg) 
                rc_train_loss_2 = self.rc_lossfn(rc_logit2, rc_labels.unsqueeze(1).repeat(1,self.args.num_avg,1,1,1).view(-1, len(constant.LABEL_TO_ID))) / (rc_labels.size(0)*self.args.num_avg)
                total_loss = (ner_train_loss_2 + rc_train_loss_2) + self.args.t1_beta*(ner_train_loss_1 + rc_train_loss_1) + info_train_loss_

                self.optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.toynet.parameters(), self.args.max_grad_norm)
                self.optim.step()

                ner_train_loss1 += ner_train_loss_1.item()
                rc_train_loss1 += rc_train_loss_1.item()
                ner_train_loss2 += ner_train_loss_2.item()
                rc_train_loss2 += rc_train_loss_2.item()
                info_train_loss += info_train_loss_.item()

                if local_iter % self.args.log_iter == 0:
                    print("[*] ner train1:{:.4f}, rc_train1:{:.4f}, ner train2:{:.4f}, rc_train2:{:.4f}, info:{:.4f}".format(ner_train_loss1/local_iter, rc_train_loss1/local_iter, ner_train_loss2/local_iter, rc_train_loss2/local_iter, info_train_loss/local_iter))
                torch.cuda.empty_cache()
                
            # save loss 
            self.history['ner_train_loss1'].append(ner_train_loss1/local_iter)
            self.history['rc_train_loss1'].append(rc_train_loss1/local_iter)
            self.history['ner_train_loss2'].append(ner_train_loss2/local_iter)
            self.history['rc_train_loss2'].append(rc_train_loss2/local_iter)
            self.history['info_train_loss'].append(info_train_loss/local_iter)
            open(self.log_file, 'a').write("[{}] ner train1:{:.4f}, rc_train1:{:.4f}, ner train2:{:.4f}, rc_train2:{:.4f}, info:{:.4f}\n".format(self.global_epoch, ner_train_loss1/local_iter, rc_train_loss1/local_iter, ner_train_loss2/local_iter, rc_train_loss2/local_iter, info_train_loss/local_iter))
            
            # evaluation after every epoch    
            with torch.no_grad():
                self.test()

        print(" [*] Training Finished!")
        best_f1 = max(self.history['F1_test'])
        best_index = self.history['F1_test'].index(best_f1)
        best_p = self.history['precision_test'][best_index]
        best_r = self.history['recall_test'][best_index]
        print("[*] best result:{:.4f}, {:.4f}, {:.4f}".format(best_p, best_r, best_f1))

    def test(self, save_ckpt=True):
        self.set_mode('eval')
        ner_test_loss1, rc_test_loss1, ner_test_loss2, rc_test_loss2, info_test_loss = 0., 0., 0., 0., 0.
        local_iter = 0
        golden_nums, predict_nums, right_nums = 0, 0, 0
        for inputs, ner_labels, rc_labels in self.data_loader['test']:
            local_iter += 1
            
            inputs = [Variable(i).cuda() for i in inputs]
            ner_labels = Variable(ner_labels).cuda()
            rc_labels = Variable(rc_labels).cuda() 
            info_test_loss_, ner_logit1, rc_logit1, ner_logit2, rc_logit2  = self.toynet(inputs, 1)
            
            # loss
            ner_test_loss_1 = self.ner_lossfn(ner_logit1, ner_labels.view(-1)) / ner_labels.size(0)
            rc_test_loss_1 = self.rc_lossfn(rc_logit1, rc_labels.view(-1, len(constant.LABEL_TO_ID))) / rc_labels.size(0)
            ner_test_loss_2 = self.ner_lossfn(ner_logit2, ner_labels.view(-1)) / ner_labels.size(0)
            rc_test_loss_2 = self.rc_lossfn(rc_logit2, rc_labels.view(-1, len(constant.LABEL_TO_ID))) / rc_labels.size(0)
            ner_test_loss1 += ner_test_loss_1.item()
            rc_test_loss1 += rc_test_loss_1.item()
            ner_test_loss2 += ner_test_loss_2.item()
            rc_test_loss2 += rc_test_loss_2.item()
            info_test_loss += info_test_loss_.item()
             
            # precision, recall, f1
            tmp_g, tmp_p, tmp_r = sta(rc_labels, ner_labels, rc_logit2.view(rc_labels.size(0), rc_labels.size(1), rc_labels.size(2), -1), ner_logit2.view(ner_labels.size(0), ner_labels.size(1), -1))
            golden_nums += tmp_g
            predict_nums += tmp_p
            right_nums += tmp_r
            torch.cuda.empty_cache()
        
        if predict_nums == 0:
            P = 0.
        else:
            P = float(right_nums) / predict_nums
        R = float(right_nums) / golden_nums
        if P+R == 0:
            F1 = 0.
        else:
            F1 = 2*P*R/(P+R)

        # save loss info
        self.history['ner_test_loss1'].append(ner_test_loss1/local_iter)
        self.history['rc_test_loss1'].append(rc_test_loss1/local_iter)
        self.history['ner_test_loss2'].append(ner_test_loss2/local_iter)
        self.history['rc_test_loss2'].append(rc_test_loss2/local_iter)
        self.history['info_test_loss'].append(info_test_loss/local_iter)    
        self.history['precision_test'].append(P)
        self.history['recall_test'].append(R)
        self.history['F1_test'].append(F1)
        print("[{}] ner test1:{:.4f}, rc test1:{:.4f}, ner test2:{:.4f}, rc test2:{:.4f}, info:{:.4f}\n".format(self.global_epoch, ner_test_loss1/local_iter, rc_test_loss1/local_iter, ner_test_loss2/local_iter, rc_test_loss2/local_iter, info_test_loss/local_iter))
        print("[{}] precision:{:.4f}, recall:{:.4f}, f1:{:.4f}\n".format(self.global_epoch, P, R, F1))
        open(self.log_file, 'a').write("[{}] ner test1:{:.4f}, rc test1:{:.4f}, ner test2:{:.4f}, rc test2:{:.4f}, info:{:.4f}\n".format(self.global_epoch, ner_test_loss1/local_iter, rc_test_loss1/local_iter, ner_test_loss2/local_iter, rc_test_loss2/local_iter, info_test_loss/local_iter))
        open(self.log_file, 'a').write("[{}] precision:{:.4f}, recall:{:.4f}, f1:{:.4f}\n".format(self.global_epoch, P, R, F1))
        
        # save the best model    
        if len(self.history['F1_test']) == 0 or max(self.history['F1_test']) == F1:
            if save_ckpt: 
                self.save_checkpoint('best_f1.tar')   
                print("[*] saved best model!")     
                open(self.log_file, 'a').write("[*] saved best model!\n")

        self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.toynet.state_dict(),
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.toynet.load_state_dict(checkpoint['model_states']['net'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
