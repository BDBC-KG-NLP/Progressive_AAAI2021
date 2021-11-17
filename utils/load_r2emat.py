import pandas as pd
import torch
from torch.autograd import Variable


def load_mat(name, t='e1'):
    tmp_df = pd.read_csv('./dataset/'+name+'/'+name+"_"+t+".csv")
    tmp_list = [row[1:] for row in tmp_df.values.tolist()]
    r2e_mat = Variable(torch.tensor(tmp_list, dtype=torch.float32), requires_grad=False)
    return r2e_mat

if __name__ == "__main__":
    r2e1_mat = load_mat('nyt', t='e1')
    r2e2_mat = load_mat('nyt', t='e2')
    print(r2e1_mat.size())
    print(r2e2_mat.size())
    
