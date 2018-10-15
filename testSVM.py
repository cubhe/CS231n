from __future__ import absolute_import
import sys
import random
import numpy  as np
from data_utils import load_cifar10
import svm as svm
from gradient_check import grad_check_sparse
import time
from linear_classifier import LinearSVM
import matplotlib.pyplot as plt
import math

x_train,y_train,x_test,y_test=load_cifar10('cifar-10-batches-py')

print('training data shape: ',x_train.shape)
print('training labels shape: ',y_train.shape)
print('test data shape: ',x_test.shape)
print('test labels shape: ',y_test.shape)

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

print('training data shape: ',x_train.shape)
print('training labels shape: ',y_train.shape)
print('validation data shape: ',x_val.shape)
print('validation labels shape: ',y_val.shape)
print('test data shape: ',x_test.shape)
print('test labels shape: ',y_test.shape,'\n')

x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))

print('training data shape: ',x_train.shape)
print('validation data shape: ',x_val.shape)
print('test data shape: ',x_test.shape)
print('development data shape: ',x_dev.shape,'\n')

mean_image=np.mean(x_train,axis=0)
x_train-=mean_image
x_val-=mean_image
x_test-=mean_image
x_dev-=mean_image

x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])
x_val=np.hstack([x_val,np.ones((x_val.shape[0],1))])
x_test=np.hstack( [x_test,np.ones((x_test.shape[0],1))] )
x_dev=np.hstack([x_dev,np.ones((x_dev.shape[0],1))])

print('training data shape: ',x_train.shape)
print('validation data shape: ',x_val.shape)
print('test data shape: ',x_test.shape)
print('development data shape: ',x_dev.shape,'\n')


w=np.random.randn(3073,10)*0.0001
loss,grad=svm.svm_loss_naive(w,x_dev,y_dev,1e2)
print('loss is : %f' % loss)
f=lambda w:svm.svm_loss_naive(w,x_dev,y_dev,1e2)[0]
grad_numerical=grad_check_sparse(f,w,grad)

tic=time.time()
loss_naive,grad_naive=svm.svm_loss_naive(w,x_dev,y_dev,0.00001)
toc=time.time()
print('naive loss: %e computed in %f s' % (loss_naive,toc-tic))

tic=time.time()
loss_vectorized,grad_vectorized=svm.svm_loss_vectorized(w,x_dev,y_dev,0.00001)
toc=time.time()
print('vectoried loss: %e computed in %f s' % (loss_vectorized,toc-tic))
print('difference: %f ' % (loss_naive-loss_vectorized))

svm=LinearSVM()
tic=time.time()
loss_hist=svm.train(x_train,y_train,learning_rate=1e-7, reg=5e4,num_iters=1500,verbose=True)
toc=time.time()
print('that took %f  s' % (toc-tic))

y_train_pred=svm.predict(x_train)
print('training accuracy: %f ' % (np.mean(y_train==y_train_pred)))
y_val_pred=svm.predict(x_val)
print('validation accuracy : %f '% (np.mean(y_val==y_val_pred)))

learning_rates=[1.4e-7,1.5e-7,1.6e-7]
regularization_strengths=[(1+i*0.1)*1e4 for i in range(-3,3)]+[(2+0.1*i)*1e4 for i in range(-3,3)]
results={}
best_val=-1
best_svm=None
for learning in learning_rates:
    for regularization in regularization_strengths:
        svm=LinearSVM()
        svm.train(x_train,y_train,learning_rate=learning,reg=regularization,num_iters=2000)
        y_train_pred=svm.predict(x_train)
        train_accuracy=np.mean(y_train==y_train_pred)
        print('training accuracy: %f ' %(train_accuracy))
        y_val_pred=svm.predict(x_val)
        val_accuracy=np.mean(best_val==y_val_pred)
        print('validation accuracy : %f ' % (val_accuracy))

        if val_accuracy>best_val:
            best_val=val_accuracy
            best_svm=svm
        results[(learning,regularization)]=(train_accuracy,val_accuracy)
for Ir ,reg in sorted(results):
    train_accuracy,val_accuracy=results[(Ir,reg)]
    print('Ir %e reg %e train accuracy : %f val accuracy : %f ' % (Ir, reg,train_accuracy,val_accuracy))
print('best validation accuracy achieved during across-validation: %f ' % best_val)


x_scatter=[math.log10(x[0]) for x in results]
y_scatter=[math.log10(x[0]) for x in results]
sz=[results[x][0]*1500 for x in results]
plt.subplot(1,2,1)
plt.scatter(x_scatter,y_scatter,sz)
plt.xlabel('log learning rate ')
plt.ylabel('log regularization strength ')
plt.title('cifar10 training accuracy')

sz=[results[x][1]*1500 for x in results]
plt.subplot(1,2,2)
plt.scatter(x_scatter,y_scatter,sz)
plt.xlabel('log learning rate ')
plt.ylabel('log regularization strength ')
plt.title('cifar10 training accuracy')

plt.show()

y_test_pred=best_svm.predict(x_test)
test_accuracy=np.mean_image(y_test==y_test_pred)
print('linear svm on raw pixels final test set accuracy: %f ' % test_accuracy)

w=bestbest_svm.W[:-1,:]
w=w.reshape(32,32,3,10)
x_min,w_max=np.min(w),np.max(w)
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck','']
for i in range(10):
    plt.subplot(2,5,i+1)
    wing=255.0*(w[:,:,:,i].squeeze()-wim)/(w_max-w_min)
    plt.imshow(wing.astype('unit8'))
    plt.axis('off')
    plt.title(classes[i])
