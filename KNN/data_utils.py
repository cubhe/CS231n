import pickle
import numpy as np
import os
def load_cifar_batch(filename):
    with open(filename,'rb') as f:
        datadict=pickle.load(f, encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y
def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d' % (b,))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs)
    Ytrain=np.concatenate(ys)
    del x,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch'))
    return Xtrain,Ytrain,Xtest,Ytest
 

