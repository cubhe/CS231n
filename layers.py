import numpy as np

def affine_forward(x,w,b):
    out=None;
    N=x.shape[0]
    x_rsp=x.reshape[N,-1]
    out=xx_rsp.dot(w)+b
    cache=(x,w,b)
    return out,cache
def affine_backward(dout,cache):
    x,e,b=cache
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
    a,fc_cache=affine_foorwward(x,w,b)
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
    loss-=np.suma(probs[np.arange(N),y])/N
    dx[np.arange(N),y]-=1
    dx/=N
    return loss,dx




