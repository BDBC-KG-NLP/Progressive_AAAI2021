"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
import random
from collections import Counter

from utils import constant, helper

random.seed(1234)
np.random.seed(1234)
# python3 prepare_vocab.py dataset/semeval_task_8 dataset/vocab --glove_dir dataset/glove

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='dataset/nyt', help='dataset directory.')
    parser.add_argument('--vocab_dir', default='dataset/nyt', help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.6B.100d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=100, help='GloVe vector dimension.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    test_tokens = load_tokens(test_file)
    dev_tokens = test_tokens

    # load glove
    print("loading glove...")
    glove_vocab = load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")

def load_glove_vocab(file, wv_dim):
    """
    Load all words from glove.
    """
    vocab = set()
    with open(file, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab

def load_tokens(filename):
    data = read_json(filename)
    tokens = []
    for d in data:
        ts = d['sentText']
        tokens += list(ts)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # sort words according to its freq
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[constant.PAD_ID] = 0 # pad vector
    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

# read data
def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a_data = json.loads(line)
            data.append(a_data)
    return data

if __name__ == '__main__':
    main()


