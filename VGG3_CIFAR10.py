import time
import torch
import torchvision 
import torchvision.transforms as transforms

transform=transforms.Compose([
        transforms.ToTensor(),#将像素值映射到[0，1.0]
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#正则化
        ])
#字面意思 数据在root文件 、训练集、从网上下载有的话就不下、transform
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
#transet 数据集、batch_size每个batch几个数据、shuffle 每次打乱
#num_works 线程 建议Windows用户不要调用多线程 我每次都报错
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
#测试集和训练集类似
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

#以上就完成了CIFAR10数据的载入

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()#调用父类的构造函数
        self.conv1=nn.Conv2d(3,64,3,padding=(1,1))#输入通道数 输出通道数 卷积核大小
        self.conv2=nn.Conv2d(64,64,3,padding=(1,1))#输入通道数 输出通道数 卷积核大小
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3,padding=(1,1))
        self.conv4=nn.Conv2d(128,128,3,padding=(1,1))   
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(128,256,3,padding=(1,1))
        self.conv6=nn.Conv2d(256,256,3,padding=(1,1))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(2,2)#大小、步长
        self.fc1=nn.Linear(256*16,512)
        self.fc2=nn.Linear(512,64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=self.conv2(self.conv1(x))#32
        x = self.bn1(x)
        x=self.pool(F.relu(x))#  32->pool->16
        x=self.conv4(self.conv3(x))
        x=self.pool(F.relu(x))#  16->pool->8        
        x=self.conv6(self.conv5(x))
        x=self.pool(F.relu(x))#  8->pool->4
        x=x.view(-1,256*4*4)#类似于numpy里的reshape 也可以用flatten
        x=self.dropout(x)
        x=F.relu(self.fc1(x))#fully conected layer
        x=self.dropout(x)
        x=F.relu(self.fc2(x))
        x=self.dropout(x)
        x=self.fc3(x)
        return x
cpu=torch.device("cpu")
gpu=torch.device("cuda")
net=Net().to(gpu)

#以上完成的了网络结构的搭建
criterion=nn.CrossEntropyLoss()#一种计算loss的方式
optimizer=torch.optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)#一种优化方式
def train(model,device,trainloader,optimizer,epoch):#训练函数
    model.train() #设置网络模式 对dropout BatchNorm有效
    for batch_idx,(data,target) in enumerate(trainloader): 
        data,target=data.to(device),target.to(device)#数据CPU->GPU
        optimizer.zero_grad()#梯度清零
        output=model(data)#输入数据
        loss=criterion(output,target)#计算损失 此处有很多方法
        loss.backward()#计算梯度
        optimizer.step()#更新梯度 grads+=lr*grads
        if(batch_idx+1)%2500==0:#输出一下免得太无聊
            print('Train Epoch: {} [ {}/{}({:.2f}%)]\tLoss:{:.6f}  '.format(\
                epoch,batch_idx+1,len(trainloader),100.*batch_idx/len(trainloader)+0.01,loss.item()))
def test(model,device,testloader):#测试函数
    model.eval()#设置网络模式 对dropout BatchNorm有效
    test_loss,correct=0,0
    with torch.no_grad():
        for data,target in testloader:
            data,target=data.to(device),target.to(device)#CPU->GPU
            output=model(data)#输入数据
            #test_loss+=F.nll_loss(output,target,reduction='sum').item()
            test_loss+=criterion(output,target)#累计损失
            pred=output.max(1,keepdim=True)[1]#找到最大的值
            correct+=pred.eq(target.view_as(pred)).sum().item()#计算算对的个数
        test_loss/=len(testloader.dataset)#loss均值
        print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(\
            test_loss,correct,len(testloader.dataset),100*correct/len(testloader.dataset)))

EPOCHS=200#20个周期
tic=time.time()#起始时间
for epoch in range(1,EPOCHS+1):
    train(net,gpu,trainloader,optimizer,epoch)
    test(net,gpu,testloader)
toc=time.time()#结束时间
print("using %ds with GPU" %(toc-tic))
