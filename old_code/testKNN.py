import numpy as np
import matplotlib.pyplot as plt
import time
from data_utils import load_cifar10
from KNN import KNearestNeighbor
x_train,y_train,x_test,y_test=load_cifar10('cifar-10-batches-py')

print('training data shape: ',x_train.shape)
print('training labels shape: ',y_train.shape)
print('test data shape: ',x_test.shape)
print('test labels shape: ',y_test.shape)


classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes=len(classes)
samples_per_class=7
for y, cls in enumerate(classes) :
    idxs=np.flatnonzero(y_train==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i , idx, in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()

num_training=5000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]

num_test=500
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,x_test.shape)

classifier=KNearestNeighbor()
classifier.train(x_train,y_train)
dists=classifier.compute_distances_two_loops(x_test)
print(dists)

y_test_pred=classifier.predict_labels(dists,k=1)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print(' got %d / %d correct => accuracy: %f  ' % (num_correct,num_test,accuracy))

dists_one=classifier.compute_distances_one_loops(x_test)
difference=np.linalg.norm(dists-dists_one,ord='fro')
print('difference was : %f ' %  difference)

dists_no=classifier.compute_distances_no_loops(x_test)
difference_no=np.linalg.norm(dists-dists_no,ord='fro')
print('difference between two and no was : %f ' % difference_no)

#验证每个方法所需要的时间
def time_function(f,*args):
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic
two_loop_time=time_function(classifier.compute_distances_two_loops,x_test)
one_loop_time=time_function(classifier.compute_distances_one_loops,x_test)
no_loop_time=time_function(classifier.compute_distances_no_loops,x_test)
print('two loops version took %f seconds ' % two_loop_time)
print('one loops version took %f seconds ' % one_loop_time)
print('no loops version took %f seconds ' % no_loop_time)


#交叉验证

num_flods=5
k_choices=[1,3,5,8,10,12,15,20,50,100]
x_train_flods=[]
y_train_flods=[]

y_train=y_train.reshape(-1,1)
x_train_flods=np.array_split(x_train,num_flods)
y_train_flods=np.array_split(y_train,num_flods)

k_to_accuracies={}

for k in k_choices:
    k_to_accuracies.setdefault(k,[])
for i in range(num_flods) :
    classifier2=KNearestNeighbor()
    x_val_train=np.vstack(x_train_flods[0:i]+x_train_flods[i+1:])
    y_val_train=np.vstack(y_train_flods[0:i]+y_train_flods[i+1:])
    y_val_train=y_val_train[:,0]
    classifier2.train(x_val_train,y_val_train)
    for k in k_choices:
        y_val_pred=classifier2.predict(x_train_flods[i],k=k)
        num_correct=np.sum(y_val_pred==y_train_flods[i][:,0])
        accuracy=float(num_correct)/len(y_val_pred)
        k_to_accuracies[k]=k_to_accuracies[k]+[accuracy]
for k in sorted(k_to_accuracies):
    sum_accuracy=0
    for accuracy in k_to_accuracies[k]:
        print('k=%d, accuracy=%f ' % (k,accuracy) )
        sum_accuracy+=accuracy
    print(' the average acuuracy is : %f ' % (sum_accuracy/5) )












