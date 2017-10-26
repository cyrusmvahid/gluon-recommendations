import os
import mxnet as mx
from mxnet import gluon, nd, ndarray

import pandas as pd
import numpy as np

data_path = '/home/ubuntu/mxnet-the-straight-dope/incubator-mxnet/example/recommenders/ml-100k/'
num_emb = 64
opt = 'Adam'
lr = 0.01
mmntm = 0.
wd = 0.
batch_size = 64
ctx = mx.gpu()

def download_ml_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Downloading MovieLens data: %s" % prefix)
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)    

def max_id(fname):
    mu = 0
    mi = 0
    with open(fname) as f:
        for line in f:
            tks = line.strip().split('\t')
            if len(tks) != 4:
                continue
            mu = max(mu, int(tks[0]))
            mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1
max_users, max_items = max_id(data_path + 'u.data')

train_df = pd.read_csv(data_path+'u1.base', header=None, sep='\t')
test_df = pd.read_csv(data_path+'u1.test', header=None, sep='\t')

train_data = nd.array(train_df[[0,1]].values, dtype=np.float32)
train_label = nd.array(train_df[2].values, dtype=np.float32)

test_data = nd.array(test_df[[0,1]].values, dtype=np.float32)
test_label = nd.array(test_df[2].values, dtype=np.float32)

class SparseMatrixDataset(gluon.data.Dataset):
    def __init__(self, data, label):
        assert data.shape[0] == len(label)
        self.data = data
        self.label = label
        if isinstance(label, ndarray.NDArray) and len(label.shape) == 1:
            self._label = label.asnumpy()
        else:
            self._label = label       
        
    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1], self.label[idx]
    
    def __len__(self):
        return self.data.shape[0]
        

class MFBlock(gluon.Block):
    def __init__(self, max_users, max_items, num_emb, dropout_p=0.5):
        super(MFBlock, self).__init__()
        
        self.max_users = max_users
        self.max_items = max_items
        self.dropout_p = dropout_p
        self.num_emb = num_emb
        
        with self.name_scope():
            self.user_biases = gluon.nn.Embedding(max_users, 1)
            self.item_biases = gluon.nn.Embedding(max_items, 1)
            self.user_embeddings = gluon.nn.Embedding(max_users, num_emb)
            self.item_embeddings = gluon.nn.Embedding(max_items, num_emb)
            self.dropout = gluon.nn.Dropout(dropout_p)
            
    def forward(self, users, items):
#        predictions = self.user_biases(users)
        
#        predictions += self.item_biases(items)
        
    
        a = self.user_embeddings(users)
        b = self.item_embeddings(items)
        predictions = a * b
        
        predictions = nd.sum(predictions, axis=1)
        return predictions

        

net = MFBlock(max_users=max_users, max_items=max_items, num_emb=num_emb, dropout_p=0.)
net.collect_params()

loss_function = gluon.loss.L2Loss()

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)

trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': mmntm, 'wd': wd})

train_data_iter = gluon.data.DataLoader(SparseMatrixDataset(train_data, train_label), 
                                        shuffle=True, batch_size=batch_size)
test_data_iter = gluon.data.DataLoader(SparseMatrixDataset(test_data, test_label),
                                          shuffle=True, batch_size=batch_size)

def eval_net(data, net):
    acc = mx.metric.Accuracy()
    for i, (user, item, label) in enumerate(data):
        user = user.as_in_context(ctx).reshape((64,))
        item = item.as_in_context(ctx).reshape((64,))
        label = label.as_in_context(ctx).reshape((64,))

        output = net(user, item)
        loss = loss_function(output, label)
        print(loss.shape)
        predictions = nd.argmax(loss)
        #acc.update(preds=predictions, labels=label)

    return acc.get()[1]
        
   
eval_net(train_data_iter, net)


epochs = 10
smoothing_constant = 0.01

def train(data_iter, net):
    for e in range(epochs):
        print("epoc: {}".format(e))
        for i, (user, item, label) in enumerate(train_data_iter):
            user = user.as_in_context(ctx).reshape((64,))
            item = item.as_in_context(ctx).reshape((64,))
            label = label.as_in_context(ctx).reshape((64,))
            with mx.autograd.record():
                output = net(user, item)               
                loss = loss_function(output, label)
                loss.backward()
    return output

train(train_data_iter, net)