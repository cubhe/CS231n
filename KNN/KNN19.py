import numpy as np
class KNearestNeighbor:
    def __init__(self):
        pass
    def train(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X,k=1,num_loops=0):
        if num_loops==0:
            pass
        elif num_loops==1:
            pass
        elif num_loops==2:
            pass
        else :
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists,k=k)
    def compute_distances_two_loops(self,X):
        num_test=X.shape[0]                # this place X.shape is [10000,3,32,320] num_test means the number of test pictures
        num_train=self.X_train.shape[0] #this place X_train.shape is  [] num_train means the number of train pictures
        dists=np.zeros((num_test,num_train))#compute every test pictures' dist to train picture and save in a array
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j]=np.sqrt(np.sum(  (X[i,:]-self.X_train[j,:])**2   )   )    #sqrt return square root in () 
        return dists
    def compute_distances_one_loops(self,X):
        num_test=X.shape[0]                # this place X.shape is [10000,3,32,320] num_test means the number of test pictures
        num_train=self.X_train.shape[0] #this place X_train.shape is  [] num_train means the number of train pictures
        dists=np.zeros((num_test,num_train))#compute every test pictures' dist to train picture and save in a array
        for i in range(num_test):
            dist[i,:]=np.sqrt(np.sum( np.square(self.X_train-X[i,:])      ),axis=1) #square can compute all elements' square 
            # worthnotice axis=1 can add every row——
        return dists
    def compute_distances_no_loops(self,X):
        num_test=X.shape[0]                # this place X.shape is [10000,3,32,320] num_test means the number of test pictures
        num_train=self.X_train.shape[0] #this place X_train.shape is  [] num_train means the number of train pictures
        dists=np.zeros((num_test,num_train))#compute every test pictures' dist to train picture and save in a array
        test_sum=np.sum(np.square(X),axis=1)#axis=1
        train_sum=np.sum(np.square(self.X_train),axis=1)
        inner_product=np.dot(X,self.X_train.T)
        dist=np.sqrt( -2*inner_product+test_sum.reshape(-1,1)+train_sum)# minus 1 mean aotucompute
        return dists
    def predict_labels(self,dists,k=1):
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closet_y=[]
            y_indicies=np.argsort(dists[i,:],axis=0)              #
            closet_y=self.y_train[y_indicies[:k]]                     #
            y_pred[i]=np.argmax(np.bincount(closet_y))       # bincout can count every lable accour times  
            #argmax can find the max value in an array which occur the most frequent 
        return y_pred
