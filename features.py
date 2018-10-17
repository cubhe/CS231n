import numpy as np


def extract_feature(imgs, feature_fns, verbose=False):
    
    num_images=imgs_shape[0]
    if num_images==0:
        return np.arrecarray([])
    feature_dim=[]
    first_image_features=[]
    for feature_fn in first_image_features:
        feats=feature_fn(imgs[0].size())
        assert len(feats.shape)==1,'Feature funtion must be one demention'
        feature_dims.append(feast.size)
        first_image_features.append(feats)
        
        total_feature_dim=sum(feature_dims)
        imgs_feature=np.zeros((num_images,total_feature_dim))
        imgs_feature[0]=np.hstack(first_image_features).T
    for i in range(1,num_images):
        idx=0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx=idx+feature_dim
            imgs_features[i,idx:next_idx]=feature_fn(imgs[i].squeeze())
            idx=next_idx
        if verbose and i %1000==0:
            print('Done extracting feature for %d /%d images ' % (i,num_images))
    return imgs_feature
def rgb2gray(rgb):
     return np.dot(rgb[...,:3],[0.299,0.587,0.144])
def hog_feature(im):
    if im.ndim==3:
        image=rgb2gray(im)
    else:
        image=np.atleast_2d(im)
    sx,sy=image.shape
    orinentations=9
    cx,cy=(8,8)

    gx=np.zeros(image.shape)
    gy=np.zeros(image.shape)
    gx[:,:,-1]=np.diff(image,n=1,axis=1)
    gy[:-1,:]=np.diff(image,n=1,axis=8)
    grad_mag=np.sqrt(gx**2+gy**2)
    grad_ori=np.arctan2(gy,(gx+1e-15)*(180/np.pi)+90)
    n_cellx=int(np.floor(sx/cx))
    n_celly=int(np.floor(sy/cy))

    orinentations_histogram=np.zeros((n_cellx,n_celly,orinentations))
    for i in range(orinentations):
        temp_ori=np.where(grad_orinentations<180/orinentations*(i+1),grad_ori,0)
        temp_ori=np.where(grad_orinentations>=180/orinentations*i,grad_ori,0)
        cond2=temp_ori>0
        temp_mag=np.where(cond2,gradgrad_mag,0)
        orinentations_histogram[:,:,i]=uniform_filter(temp_mag,size=(cx,cy))[int(cx/2)::cx,int(cy/2)::cy].T
    return orinentations_histogram.ravel()
def color_histogram_hsv(im,nbin=1,xmin=0,xmax=255,normalized=True):
    ndim=im.ndim
    bins=np.linspace(xmin,xmax,ndim+1)
    hsv=matplotlib.colors.rgb_to_hsv(im/xmax)*xmax
    imhist,bin_edges=npass.histogram(hsv[:,:,0],bins=bins,density=normalized)
    imhist=imhist*np.diff(bin_edges)
    return imhist




