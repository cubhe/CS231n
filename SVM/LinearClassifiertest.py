from __future__ import absolute_import
import sys
sys.path.append("..")
import random 
import numpy as np
from data_utils import load_cifar10
from SVM19 import *
from softmax import *
from gradient_check import grad_check_sparse
from linear_classifier import LinearSVM
import time

cifar10_dir='cifar-10-batches-py'
x_train,y_train,x_test,y_test=load_cifar10(cifar10_dir)
#抽样
num_training=49000
num_validation=1000
num_test=1000
num_dev=500
mask=range(num_training,num_training+num_validation)
x_val=x_train[mask]
y_val=y_train[mask]
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
mask=np.random.choice(num_training,num_dev,replace=False)
x_dev=x_train[mask]
y_dev=y_train[mask]
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]
#print
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_dev.shape,y_dev.shape,x_test.shape,y_test.shape)
#拉成一维
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_dev.shape,y_dev.shape,x_test.shape,y_test.shape)
#归一化
mean_image=np.mean(x_train,axis=0)
x_train-=mean_image
x_val-=mean_image
x_dev-=mean_image
x_test-=mean_image
#给矩阵加一列
x_train=np.hstack(  [x_train,np.ones((x_train.shape[0],1))])
x_val=np.hstack(  [x_val,np.ones((x_val.shape[0],1))])
x_dev=np.hstack(  [x_dev,np.ones((x_dev.shape[0],1))])
x_test=np.hstack(  [x_test,np.ones((x_test.shape[0],1))])
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_dev.shape,y_dev.shape,x_test.shape,y_test.shape)
#main code
'''
#test svm naive
w=np.random.randn(3073,10)*0.0001
loss,grad=svm_loss_naive(w,x_dev,y_dev,1e2)
f= lambda w:svm_loss_naive(w,x_dev,y_dev,1e2)[0]
grad_numerical=grad_check_sparse(f,w,grad)
print('loss is : %f' %loss)
'''
'''
#test svm linearclassifier
svm=LinearSVM()
tic=time.time()
loss_hist=svm.train(x_train,y_train,learning_rate=1e-7,reg=5e4,num_iters=2500,verbose=True)
toc=time.time()
print('that took %f s' %(toc-tic))
y_train_pred=svm.predict(x_train)
y_val_pred=svm.predict(x_val)
print('training accuracy: %f ' %(np.mean(y_train==y_train_pred)))
print('validation accuracy: %f ' %(np.mean(y_val==y_val_pred)))
'''
#test softmax naive
w=np.random.randn(3073,10)*0.0001
loss,grad=softmax_loss_naive(w,x_dev,y_dev,1e3)
f= lambda w:softmax_loss_naive(w,x_dev,y_dev,1e3)[0]
grad_numerical=grad_check_sparse(f,w,grad)
print('loss is : %f' %loss)