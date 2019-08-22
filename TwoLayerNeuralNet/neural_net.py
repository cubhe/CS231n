import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,std=1e-4):         #初始化这两个层的参数，用一个很小的值 偏置量为0
        self.params={}
        self.params['W1']=std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
    def loss(self,x,y=None,reg=0.0):
        W1,b1=self.params['W1'],self.params['b1']#这一部分就是softmax 无区别
        W2,b2=self.params['W2'],self.params['b2']
        N=x.shape[0]
        D=x.shape[1]
        scores=None
        h_output=np.maximum(0,x.dot(W1)+b1)
        scores=h_output.dot(W2)+b2
        if y is None:
            return scores
        loss= None
        shift_scores=scores-np.max(scores,axis=1).reshape((-1,1))
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape((-1,1))
        loss=-np.sum(np.log(softmax_output[range(N),list(y)]))
        loss/=N #平均数
        loss+=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))#正则化
        #计算梯度时要算俩
        grads={}
        dscores=softmax_output.copy()
        dscores[range(N),list(y)]-=1
        dscores/=N
        grads['W2']=h_output.T.dot(dscores)+reg*W2
        grads['b2']=np.sum(dscores,axis=0)

        dh=dscores.dot(W2.T)
        dh_Relu=(h_output>0)*dh
        grads['W1']=x.T.dot(dh_Relu)+reg*W1
        grads['b1']=np.sum(dh_Relu,axis=0)

        return loss,grads
    def train(self,x,y,x_val,y_val,learning_rate=1e-3,learning_rate_decay=0.95,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train=x.shape[0]
        iterations_per_epoch=max(num_train/batch_size,1)
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        for it in range(num_iters):
            x_batch=None
            y_batch=None
            idx=np.random.choice(num_train,batch_size,replace=True)
            x_batch=x[idx]
            y_batch=y[idx]
            loss,grads=self.loss(x_batch,y_batch,reg=reg)
            loss_history.append(loss)

            self.params['W2']+=-learning_rate*grads['W2']
            self.params['b2']+=-learning_rate*grads['b2']
            self.params['W1']+=-learning_rate*grads['W1']
            self.params['b1']+=-learning_rate*grads['b1']

            if verbose and it %100==0:
                print('iteration %d/%d : loss %f' %(it  ,num_iters,loss))
            if it %iterations_per_epoch==0:
                train_acc=(self.predict(x_batch)==y_batch).mean()
                val_acc=(self.predict(x_val)==y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate*=learning_rate_decay#带动量的SGD
        return {'loss_history':loss_history,'train_acc_history':train_acc_history,'val_acc_history':val_acc_history}
    def predict(self,x):
        y_pred=None
        h=np.maximum(0,x.dot(self.params['W1'])+self.params['b1'])
        scores=h.dot(self.params['W2'])+self.params['b2']
        y_pred=np.argmax(scores,axis=1)
        return y_pred
