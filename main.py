import numpy as np
import torch
import argparse
from solver import Solver as Solver


def main(args):

    seed = args.seed
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()
    
    net = Solver(args)

    if args.mode == 'train' : net.train()
    elif args.mode == 'test' : net.test(save_ckpt=False)
    else : return 0

if __name__ == "__main__":
    # training with GPU

    parser = argparse.ArgumentParser(description='Multi task VIB')
    parser.add_argument('--log_iter', default = 100, type=int, help='interval of printing loss info' )
    parser.add_argument('--log_file', default = 'log.txt', type=str, help='file to save loss info')
    parser.add_argument('--lr', default = 0.001, type=float, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--max_sent_len', type=int, default=300)
    parser.add_argument('--lower', default=False, help='Lowercase all words.')
    parser.add_argument('--K', default = 100, type=int, help='dimension of encoding Z')
    parser.add_argument('--rnn_hidden', default = 50, type=int, help='dimension of rnn layer')
    parser.add_argument('--emb_dim', type=int, default=100, help='Word embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=10, help='POS embedding dimension.')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--batch_size', default = 50, type=int, help='batch size')
    parser.add_argument('--dset_dir', default='dataset', type=str, help='dataset directory path')
    parser.add_argument('--mode',default='train', type=str, help='train or eval')
    parser.add_argument('--input_dropout', type=float, default=0.1, help='input dropout rate.')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    #nyt
    parser.add_argument('--epoch', default = 100, type=int, help='epoch size')
    parser.add_argument('--t1_beta', default = 1e-3, type=float, help='beta')
    parser.add_argument('--beta1', default = 1e-6, type=float, help='beta')
    parser.add_argument('--beta2', default = 1e-6, type=float, help='beta')
    parser.add_argument('--dataset', default='nyt', type=str, help='dataset name')
    parser.add_argument('--num_avg', default = 1, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)
