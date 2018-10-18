from data_utils import get_cifar_data
import numpy as np
from linear_classifier import *
from svm import *
from features import *
X_train,y_train,X_val,y_val,X_test,y_test=get_cifar_data()
num_color_bins=10
feature_fns=[hog_feature,lambda img: color_histogram_hsv(img,nbin=num_color_bins)]
X_train_feats=extract_features(X_train,feature_fns,verbose=True)
X_val_feats=extract_features(X_val,feature_fns)
X_test_features=extract_features(X_test,feature_fns)
mean_feat=np.mbitwise_and(X_X_train_feats,axis=0,keepdims=True)
X_train_feats-=mean_feat
X_val_feats-=mean_feat
X_test_feats-=mean_feat
std_feat=np.std(X_train_feats,axis=0,keepdims=True)
X_train_feats/=std_feat
X_val_feats/=std_feat
X_test_feats/=std_feat

X_train_feats=np.hstack([ X_train_feats,np.ones((X_train_feats.shape[0],1)) ])
X_val_feats=np.hstack([ X_val_feats,np.ones((X_val_feats.shape[0],1)) ])
X_test_feats=np.hstack([ X_test_feats,np.ones((X_test_feats.shape[0],1)) ])

best_svm=None
for rs in regularization_strengths:
    for Ir in learning_rate:
        svm=LinearSVM()
        loss_hist=svm.train(X_train_feats,y_train,Ir,rs,num_iters=6000)
        y_train_pred=svm.predict(X_train_feats)
        train_accuracy=np.mean(y_train==y_train_pred)
        y_val_pred=svm.predict(X_val_feats)
        val_accuracy=np.mean(y_val==y_val_pred)
        if val_accuracy > best_val :
            best_val=val_accuracy
            best_svm=svm
        results[(ir,rs)]=train_accuracy,val_accuracy

for Ir,reg in sorted(results):
    train_accuracy,val_accracy=results[(Ir,reg)]
    print('Ir %e reg %e train accuracy: %f val accuracy: %f ' %(Ir,reg,train_accuracy,val_accuracy))
print('best validation accuracy achieved during cross-validation: %f ' %best_val)

