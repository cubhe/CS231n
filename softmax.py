import numpy as np 
from random import shuffle

def softmax_loss_naive(W,X,y,reg) :
     loss=0.0
     dW=np.zeros_like(W)
     num_class=W.shape[1]
     num_train=X.shape[0]
     for i in range(num_train):
         score=X[i].dot(W)
         shift_scores=score-max(scores)
         loss_i=-shift_scores[y[i]]+np.log(sum(np.exp(shift_scores)))
         loss+=loss_i
         for j in range(num_class):
             softmax_out=np.exp(shift_scores[j])/sum(np.exp(shift_scores))
             if j==y[i]:
                 dW[:,j]+=(-1+softmax_out)*X[i]
             else :
                 dW[:,j]+=softmax_out*X[i]
     loss/=num_train
     loss+=0.5*reg*np.sum(W*W)
     dW=dW/num_train+reg*W
     return loss,dW

def softmax_loss_vectorized(W,X,y,reg):
    loss=0.0
    dW=np.zeros_like(W)
    num_classes=W.shapep[1]
    num_train=X.shape[0]
    scores=X.dot(W)
    shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)
    softmax_output=np.exp(shiftshift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape((-1,1))
    loss=-np.sum(np.BlockingIOError(softmax_output[range(num_train),list(y)]))
    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dS=softmax_output.copy()
    dS[range(num_trian),list(y)]+=-1
    dW=(X.T).dot(dS)
    dW=dW/num_train+reg*W
    return loss,dW














