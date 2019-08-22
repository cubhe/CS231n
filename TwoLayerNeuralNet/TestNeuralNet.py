from data_utils import load_cifar10
from neural_net import TwoLayerNet
import matplotlib.pyplot as plt
import numpy as np
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

#main code
'''
input_size=32*32*3#给定每层的神经元数量
hidden_size=50
num_classes=10
net=TwoLayerNet(input_size,hidden_size,num_classes)
stats=net.train(x_train,y_train,x_val,y_val,num_iters=1000,batch_size=200,learning_rate=1e-4,learning_rate_decay=0.95,reg=0.5,verbose=True)
val_acc=(net.predict(x_val)==y_val).mean()
print('validation accuracy :', val_acc)
'''
input_size=32*32*3
num_classes=10
hidden_size=[75,100,125]
results={}
best_val_acc=0
best_net=None
learning_rate=np.array([0.7,0.8,0.9,1.0,1.1])*1e-3
regularization_strengths=[0.75,1,1.25]
print('running')
tis=time.time()
for hs in hidden_size:
    for lr in learning_rate:
        for reg in regularization_strengths:
            net=TwoLayerNet(input_size,hs,num_classes)
            states=net.train(x_train,y_train,x_val,y_val,num_iters=1500,batch_size=200,learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
            val_acc=(net.predict(x_val)==y_val).mean()
            if val_acc>best_val_acc:
                best_val_acc=val_acc
                best_net=net
            results[(hs,lr,reg)]=val_acc
            print('hs %d lr %e reg %e val accuracy: %f ' %(hs , lr, reg , val_acc))
tie=time.time()
print('finished. total time : %f s' %(tie-tis))

for hs ,lr , reg in sorted(results):
    val_acc=results[(hs,lr,reg)]
    print('hs %d lr %e reg %e val accuracy: %f ' %(hs , lr, reg , val_acc))
    print('best validation accuracy achieved during cross_validation : %f' % best_val_acc)
#画图玩
plt.subplot(211)
plt.plot(stats['loss_history'])
plt.title('loss history')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.subplot(212)
plt.plot(stats['train_acc_history'],label='train')
plt.plot(stats['val_acc_history'],label='val')
plt.title('classification accuracy history')
plt.xlabel('epoch')
plt.ylabel('classification accuracy')
plt.show()

