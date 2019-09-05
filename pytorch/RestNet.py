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
def conv33(in_channles,out_channles,stride=1):
    return nn.Conv2d(in_channles,out_channles,kernel_size=3,stride=stride,padding=1,bias=False)
class BasicBlock(nn.Module):
    def __init__(self,in_channles,out_channles,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv33(in_channles,out_channles,stride)
        self.bn1=nn.BatchNorm2d(out_channles)
        self.relu=nn.ReLU(True)
        self.conv2=conv33(out_channles,out_channles)
        self.bn2=nn.BatchNorm2d(out_channles)
        self.downsample=downsample
    def forward(self,x):
        residual=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        '''print(x.shape)
        print(out.shape)
        print(residual.shape)
        print(' ')'''
        out+=residual
        out=self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channles=16
        self.conv=conv33(3,16)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(True)
        self.layer1=self.make_layer(block,16,layers[0])
        self.layer2=self.make_layer(block,32,layers[0],2)
        self.layer3=self.make_layer(block,64,layers[1],2)
        self.avg_pool=nn.AvgPool2d(8)
        self.fc=nn.Linear(64,num_classes)
    def make_layer(self,block,out_channles,blocks,stride=1):
        downsample=None
        if out_channles != self.in_channles or stride !=1:
            downsample = nn.Sequential(conv33(self.in_channles,out_channles,stride=stride),nn.BatchNorm2d(out_channles))
        layers=[]
        layers.append(block(self.in_channles,out_channles,stride,downsample))
        self.in_channles=out_channles
        for i in range(1,blocks):
            layers.append(block(out_channles,out_channles))
        return nn.Sequential(*layers)
    def forward(self,x):
        #print(x.shape)
        out=self.relu(self.bn(self.conv(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        #print(out.shape)
        out=self.layer3(out)
        #print(out.shape)
        out = self.avg_pool(out)
        out=self.fc(out.view(out.size(0),-1))
        return out
cpu=torch.device("cpu")
gpu=torch.device("cuda")
net=ResNet(BasicBlock,[3,4]).to(gpu)

#以上是网络构建
criterion=nn.CrossEntropyLoss()#一种计算loss的方式
optimizer=torch.optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)#一种优化方式
#schedule_lr = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
print("using %ds 'with GPU" %(toc-tic))