import numpy as np
from random import shuffle

def svm_loss_naive(w,x,y,reg):#计算损失值和梯度导数
    dw=np.zeros(w.shape)
    num_classes=w.shape[1]     #有多少种图片
    num_train=x.shape[0]        #有多少张图片
    loss=0.0
    for i in range(num_train):
        scores=x[i].dot(w)            #点乘 1*3073.3073*10=1*10
        correct_class_score=scores[y[i]] #score[yi]]对应的就是正确数据 注意此处数据标签只能按顺序来
        for j in range(num_classes):
            if j==y[i]:                     #跳过目标值
                continue
            margin=scores[j]-correct_class_score+1 #计算损失值 其他分类减去正确分类 
            if margin>0:                                          #如果正确分类分数较高的话那么margin应该为负则loss为0
                loss+=margin                                     #如果为正说明分类错误 累加loss
                dw[:,j]+=x[i].T                                    #dw是什么一直没有搞懂 我去看看线性分类器再回来
                dw[:,y[i]]+=-x[i].T                                #我看完了 原来dw是梯度 X是1*3073 W是3073*10 所以要先transposition一下
    loss /=num_train                                           #计算平均值
    dw /= num_train                                           #c此处不是均值 和导数定义有关
    loss+=0.5*reg*np.sum(w*w)                           #此处迷惑 就是加一个偏置量吧
    dw+=reg*w                                                     #此处迷惑 又是一个偏置量
    return loss,dw
def svm_loss_vectorized(w,x,y,reg):                     #这个和naive是一样的操作 只是用矩阵的形式表达出来了
    num_train=x.shape[0]                                     #这几步同上
    num_classes=w.shape[1]
    scores=x.dot(w)
    correct_class_scores=scores[range(num_train),list(y)].reshape(-1,1)#
    margins=np.maximum(0,scores-correct_class_scores+1)#相当于判断得分是否为负
    margins[range(num_train),list(y)]=0                          
    loss=np.sum(margins)/num_train+0.5*reg*np.sum(w*w)
    coeff_mat=np.zeros((num_train,num_classes))
    coeff_mat[margins>0]=1
    coeff_mat[range(num_train),list(y)]=0
    coeff_mat[range(num_train),list(y)]=-np.sum(coeff_mat,axis=1)
    dw=(x.T).dot(coeff_mat)
    dw=dw/num_train+reg*w
    return loss,dw