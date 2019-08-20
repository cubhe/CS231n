import numpy as np

def affine_forward(x,w,b):
    out=None;
    N=x.shape[0]
    x_rsp=x.reshape(N,-1)
    out=x_rsp.dot(w)+b
    cache=(x,w,b)
    return out,cache
def affine_backward(dout,cache):
    x,w,b=cache
    dx,dw,db=None, None, None
    N=x.shape[0]
    x_rsp=x.reshape(N,-1)
    dx=dout.dot(w.T)
    dx=dx.reshape(*x.shape)
    dw=x_rsp.T.dot(dout)
    db=np.sum(dout,axis=0)
    return dx,dw,db
def relu_forward(x):
    out=None
    out=x*(x>=0)
    cache=x
    return out,cache
def relu_backward(dout,cache):
    dx,x=None,cache
    dx=(x>=0)*dout
    return dx
def affine_relu_forward(x,w,b):
    a,fc_cache=affine_forward(x,w,b)
    out,relu_cache=relu_forward(a)
    cache=(fc_cache,relu_cache)
    return out,cache
def affine_relu_backward(dout,cache):
    fc_cache,relu_cache=cache
    da=relu_backward(dout,relu_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db
def svm_loss(x,y):
    N=x.shape[0]
    correct_class_scores=x[np.arange(N),y]
    margins=np.maximum(0,x-correct_class_scores[:,np.newaxis]+1.0)
    margins[np.arange(N),y]=0
    loss=np.sum(margins)/N

    num_pos=np.sum(margins>0,axis=1)
    dx=np.zeros_like(x)
    dx[margins>0]=1
    dx[np.arange(N),y]-=num_pos
    dx/=N
    return loss,dx
def softmax_loss(x,y):
    probs=np.exp(x-np.max(x,axis=1,keepdims=True))
    probs/=np.sum(probs,axis=1,keepdims=True)
    N=x.shape[0]
    loss=-np.sum(np.log(probs[np.arange(N),y]))/N
    dx=probs.copy()
    dx[np.arange(N),y]-=1
    dx/=N
    return loss,dx
class TwoLayerNet(object):
    def __init__(self,input_dim=3*32*32,hidden_dim=100,num_classes=10,weight_scale=1e-3,reg=0.0):
        self.params={}
        self.reg=reg
        self.params['W1']=weight_scale*np.random.rand(input_dim,hidden_dim)
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2']=np.zeros(num_classes)
    def loss(self,X,y=None):
        scores=None
        ar1_out,ar1_cache=affine_relu_forward(X,self.params['W1'],self.params['b1'])
        a2_out,a2_cache=affine_forward(ar1_out,self.params['W2'],self.params['b2'])
        scores=a2_out
        if y is None:
            return scores
        loss,grads=0,{}
        loss,dscores=softmax_loss(scores,y)
        loss=loss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
        
        dx2,dw2,db2=affine_backward(dscores,a2_cache)
        grads['W2']=dw2+self.reg*self.params['W2']
        grads['b2']=db2

        dx1,dw1,db1=affine_relu_backward(dx2,ar1_cache)
        grads['W1']=dw1+self.reg*self.params['W1']
        grads['b1']=db1
        return loss,grads







