import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
BATCH_SIZE=512
EPOCHS=20
DEVICE=torch.device("cuda")


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
test_loader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE,shuffle=True)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1=nn.Linear(20*10*10,500)
        self.fc2=nn.Linear(500,10)
    def forward(self,x):
        in_size=x.size(0)
        out=self.conv1(x)#24
        out=F.relu(out)
        out=F.max_pool2d(out,2,2)#12
        out=self.conv2(out)#10
        out=F.relu(out)
        out=out.view(in_size,-1)
        out=self.fc1(out)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.log_softmax(out,dim=1)
        return out
model = ConvNet().to(DEVICE)
optimizer=optim.Adam(model.parameters())
def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30==0:
            print('Train Epoch: {}  [ {}/{} ({:.0f}%) ]\tLoss: {:.6f}'.format(\
                epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item())\
                )
def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target =data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.max(1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(\
            test_loss,correct,len(test_loader.dataset),100*correct/len(test_loader.dataset)))
tic=time.time()
for epoch in range(1,EPOCHS+1):
    train(model,DEVICE,train_loader,optimizer,epoch)
    test(model,DEVICE,test_loader)
toc=time.time()
print("using %d s with GPU" %(toc-tic))
