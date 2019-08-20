import numpy as np
from SVM19 import *

class LinearClassifer:
    def __init__(self):
        self.W=None
    def train(self,x,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train,dim=x.shape   #获取 输入图像的数量和维度//已经被拉长了是1*3072
        num_classes=np.max(y)+1 
        if self.W==None:#initial w by give w a litrle value
            self.W=0.001*np.random.randn(dim,num_classes)
        loss_history=[]
        for i in range(num_iters):
            x_batch=None
            y_batch=None
            batch_idx=np.random.choice(num_train,batch_size,replace=True)
            #第四个参数是概率 没有就是均匀分布。从num_train中均匀抽取200个作为输入
            x_batch=x[batch_idx]
            y_batch=y[batch_idx]
            loss,grad=self.loss(x_batch,y_batch,reg)#将随机抽取的图片输入loss_function计算出损失值 和梯度
            loss_history.append(loss)                      #记录loss
            self.W+=-1*learning_rate*grad              #叠加梯度值更新W
            if  verbose and i %  100==0:  #迭代100次输出一个提示
                print('iteration %d  / %d: loss %f' %(i, num_iters,loss))
        return loss_history
    def predict(self,x):
        y_pred=np.zeros(x.shape[1])
        scores=x.dot(self.W)
        y_pred=np.argmax(scores,axis=1)
        return y_pred
    def loss(self, X_batch ,y_batch,reg):#此处还可以用别的方式计算loss
        pass
class LinearSVM(LinearClassifer):     
    def loss(self,X_batch,y_batch,reg):
        return svm_loss_vectorized(self.W,X_batch,y_batch,reg)
