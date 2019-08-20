import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from KNN19 import KNearestNeighbor
x_train,y_train,x_test,y_test=load_cifar10('cifar-10-batches-py')

print('training data shape: ', x_train.shape)
print('training labels shape: ',y_train.shape)
print('test data shape:', x_test.shape)
print('test labels shape: ',y_test.shape)
'''
classes=['plane','car','bird','car','deer','dog','frog','horse','ship','truck']
num_classes=len(classes)
samples_per_class=7
for y , cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i , idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()
'''
#pick 5000 data in 50000
num_train=5000
mask=range(num_train)
x_train=x_train[mask]
y_train=y_train[mask]

num_test=500
mask=range(num_train)
x_test=x_test[mask]
y_test=y_test[mask]
print(x_train.shape,x_test.shape)
#reshape picture into one dimension vetor
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
print('Got %d / %d correct => accuracy: %f ' % (num_correct,num_test,accuracy))

