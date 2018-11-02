import numpy as np
import matplotlib.pyplot as plt
from Solver import Solver
from data_utils import *
from layers import TwoLayerNet
print('1')
X_train,y_train,X_val,y_val,X_test,y_test = get_cifar_data()
print('2')
data={}
print('3')
data={'X_train' : X_train, 'y_train' : y_train, 'X_val': X_val , 'y_val': y_val,'X_test':X_test,'y_test':y_test}
print('4')
model=TwoLayerNet(reg=1e-1)
solver=None
print('5')
solver=Solver(model,data,update_rule='sgd',optim_config={'learning_rate':1e-3},lr_decay=0.8,num_epochs=10,batch_size=100,print_every=100)
print('6')
solver.train()
print('7')
scores=model.loss(data['X_test'])
y_pred=np.argmax(scores,axis=1)
acc=np.mean(y_pred==data['y_test'])
print('test acc: %f' % acc)

