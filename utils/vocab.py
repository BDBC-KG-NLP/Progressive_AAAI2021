import os
import pickle

class Vocab(object):
    def __init__(self, filename):
        assert os.path.exists(filename), "Vocab file does not exist at " + filename
        # load from file and ignore all other params
        self.id2word, self.word2id = self.load(filename)
        self.size = len(self.id2word)
        print("Vocab size {} loaded from {}.".format(self.size, filename))

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id

    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
        return
