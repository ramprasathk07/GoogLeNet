import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(self,num_class=1000,inchannel=3):
        super().__init__()

        self.conv1 = conv_block(in_channel=inchannel,out_channel=64,kernel_size =(7,7),
                                stride=(2,2),padding = (3,3))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 = conv_block(in_channel=64,out_channel=192,kernel_size = 3,stride =1, 
                                padding=1)
        self.maxpool11 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a = Inception_block(192,64,96,128,16,32,32)
        self.inception3b = Inception_block(256,128,128,192,32,96,64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = Inception_block(480,192,96,208,16,48,64)
        self.inception4b = Inception_block(512,160,112,224,24,64,64) #add a avg->conv1x1+1s->FC1->FC2->softmax
        self.inception4c = Inception_block(512,128,128,256,24,64,64)
        self.inception4d = Inception_block(512,112,144,288,32,64,64)
        self.inception4e = Inception_block(528,256,160,320,32,128,128)#add a avg->conv1x1+1s->FC1->FC2->softmax

        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = Inception_block(832,256,160,320,32,128,128)
        self.inception5b = Inception_block(832,384,192,384,48,128,128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024,num_class)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool11(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool2(x)

        x = self.inception4a(x)
        x = self.inception4b(x)        
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool3(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)

        return x

class Inception_block(nn.Module):
    def __init__(self,in_channel,
                 out_1x1,
                 red_3x3,
                 out_3x3,
                 red_5x5,
                 out_5x5,out_1x1pool):
        
        super().__init__()

        self.br_1x1 = conv_block(in_channel=in_channel,
                                 out_channel=out_1x1,
                                 kernel_size = 1, )
        self.br2 = nn.Sequential(
            conv_block(in_channel,red_3x3,kernel_size = 1,),
            conv_block(red_3x3,out_3x3,kernel_size = 3,padding = 1))
        
        self.br3 = nn.Sequential(
            conv_block(in_channel,red_5x5,kernel_size = 1),
            conv_block(red_5x5,out_5x5,kernel_size = 5,padding = 2)
        )
        self.br4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            conv_block(in_channel,out_1x1pool,kernel_size =1)
        )

    def forward(self,x):
        return torch.cat([self.br_1x1(x),self.br2(x),self.br3(x),self.br4(x)],1)        

class conv_block(nn.Module):
    def __init__(self,in_channel,out_channel,**kwargs):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel,out_channel,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))
    
if __name__ =="__main__":
    x = torch.rand(20,3,224,224)
    model = GoogLeNet()
    y = model(x)
    print(y.shape)