import pickle                          #解压cifar10
import numpy as np
import os                                #开文件

'''
1.用只读二进制的方法打开filename指向的文件
2.x储存图片信息 y储存标签信息
3.字符串前＋b代表二进制
'''
def load_cifar_batch(filename):
    with open(filename,'rb') as f :
        datadict=pickle.load(f,encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y
    return 


'''
1.打开root文件夹=>打开五个data_path文件
2.将文件内容加入xs xy append函数
3.把几个x的内容合并成一个大的并用作数据集 concatenate函数
4.读取test_batch文件里的内容 并用作测试集
'''
def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f= os.path.join(root,'data_batch_%d' % (b, ))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs)
    Ytrain=np.concatenate(ys)
    del x,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch'))
    return Xtrain,Ytrain,Xtest,Ytest

