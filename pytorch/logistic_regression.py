import torch
import torch.nn as nn
import numpy as np

data=np.loadtxt("german.data-numeric")
n,l=data.shape
for i in range(l,-1):
    meaval=np.mean(data[:,j])
    stdval=np.std(data[:,j])
    data[:,j]=(data[:,j]-meaval)/stdval
np.random.shuffle(data)
train_data=data[:900,:l-1]
train_lab=data[:900,l-1]-1
test_data=data[900:,:l-1]
test_lab=data[900:,l-1]-1
print(data.shape)
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(24,2)
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)
        return out
def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())

net=LR()
criterion=nn.CrossEntropyLoss()
optm=torch.optim.Adam(net.parameters())
epochs=20000

for i in range(epochs):
    net.train()
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    #这里有点坑啊！！！这个类型一定要 搞清楚
    y_hat=net(x)
    #y_hat=y_hat.float()
    #print(y_hat.shape)
    loss=criterion(y_hat,y)
    optm.zero_grad()
    loss.backward()
    optm.step()
    if (i+1)%1000==0:
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        accu=test(test_out,test_l)
        print("Epoch:{},loss:{:.4f},accuracy:{:.2f}".format(i+1,loss.item(),accu))

