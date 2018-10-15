import numpy as np
import matplotlib.pyplot as plt
from neural_net import * 
from data_utils import *
X_train,y_train,X_val,y_val,X_test,y_test = get_cifar_data()
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size,hidden_size,num_classes)
states = net.train(X_train,y_train,X_val,y_val,num_iters=1000,batch_size=200,learning_rate=1e-4,learning_rate_decay=0.95,reg=0.5,verbose=True)
val_acc = (net.predict(X_val) == y_val).mean()
print('validation accuracy: ' ,val_acc)

#loss
plt.subplot(211)
plt.plot(states['loss_history'])
plt.title('loss history')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(states['train_acc_history'],label='train')
plt.plot(states['val_acc_history'],label='val')
plt.title('classification accuracy history')
plt.xlabel('epoch')
plt.ylabel('classification accuracy')
plt.show()

imput_size = imput_size = 32 * 32 * 3
num_classes = 10
hidden_size = [75,100,125]
results = {}
best_val_acc = 0
best_net = None
learning_rates = np.array([0.7,0.8,0.9,1.0,1.1]) * 1e-3
regularization_strengths = [0.75,1.0,1.25]
print('start')
for hs in hidden_size:
    for Ir in learning_rates:
        for reg in regularization_strengths:
            net = TwoLayerNet(input_size,hs,num_classes)
            stats = net.train(X_train,y_train,X_val,y_val,num_iters=1500,batch_size=200,learning_rate=Ir,learning_rate_decay=0.95,reg=reg,verbose=True)
            val_acc=(net.predict(X_val)==y_val).mean()
            print('validation accuracy: ' ,val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
            results[(hs,Ir,reg)] = val_acc
print('finished')
for hs,Ir,reg in sorted(results):
    val_acc = results[(hs,Ir,reg)]
    print('hs %d Ir %e reg %e val accuracy: %f' % (hs,Ir,reg,val_acc))
print('best validation accuracy achieved during cross_validation: %f ' % best_val_acc)

