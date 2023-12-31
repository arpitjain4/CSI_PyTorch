import torch
import torch.nn as nn

## Refinenet unit definiton
class REFINENETUNIT_ANN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(REFINENETUNIT_ANN, self).__init__()


        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=8, 
                                     kernel_size=3, stride=1, padding=1, dilation=1, groups=1, 
                                     bias=True, padding_mode='zeros')

        self.bn3 =nn.BatchNorm2d(8)
        self.relu=nn.LeakyReLU(negative_slope=0.3)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, 
                                     kernel_size=3, stride=1, padding=1, dilation=1, groups=1, 
                                     bias=True, padding_mode='zeros')
        self.bn4 =nn.BatchNorm2d(16)
        # self.relu4=nn.LeakyReLU(negative_slope=0.3)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=out_channels, 
                                     kernel_size=3, stride=1, padding=1, dilation=1, groups=1, 
                                     bias=True, padding_mode='zeros')
        self.bn5 =nn.BatchNorm2d(out_channels)

        self.relu5 =nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        shortcut = x
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        # # x=nn.functional.leaky_relu(nn.functional.batch_norm(self.conv1(x, stochastic=stochastic)),negative_slope=0.3)
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        # #x=nn.functional.leaky_relu(nn.functional.batch_norm(self.conv2(x, stochastic=stochastic)),negative_slope=0.3)
        x=self.conv5(x)
        x=self.bn5(x)
        # #x=nn.functional.batch_norm(self.conv3(x, stochastic=stochastic))
        x += shortcut
        x = self.relu(x)
        return x

# CSINET model defintion
class CSINET_ANN(nn.Module):
    def __init__(self,M):
        super(CSINET_ANN, self).__init__()

        self.Nc=32
        self.Nt=32
        self.M= M

        self.conv1 = nn.Conv2d( in_channels=2, out_channels=2,kernel_size=3, stride=1, padding=1, dilation=1, groups=1,bias=True, padding_mode='zeros')
        self.bn1=nn.BatchNorm2d(2)
        self.relu1=nn.LeakyReLU(negative_slope=0.3)
        self.linear1 = nn.Linear(in_features=2*self.Nc*self.Nt, out_features=self.M, bias=True)       


        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2,kernel_size=3, stride=1, padding=1, dilation=1, groups=1,bias=True, padding_mode='zeros')
        

        # Decoder layers
        self.linear2 = nn.Linear(in_features=self.M, out_features=2*self.Nc*self.Nt,bias=True)

        self.refinenet1 = REFINENETUNIT_ANN(2, 2)
        self.refinenet2 = REFINENETUNIT_ANN(2, 2)

        
        self.bn3 = nn.BatchNorm2d(2)
        self.sigmoid =nn.Sigmoid()
        #self.tanh = nn.Tanh()


    def forward(self, x):

        #encoder layer
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        #x = nn.functional.leaky_relu(nn.functional.batch_norm(self.conv1(x, stochastic=stochastic)))

        x = x.view(x.size(0), -1) #(reshaping) flatten  2 channel output for fully connected layer input
        x = self.linear1(x) #linear layer

        # Decoder layers
        x = self.linear2(x) #linear layer
        x = x.view(x.size(0), 2, self.Nc, self.Nt) #(reshaping)

        #passing through refinet unit
        x = self.refinenet1(x)
        x = self.refinenet2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x= self.sigmoid(x)
        #x=self.tanh(x)
        #x = nn.functional.sigmoid(nn.functional.batchnorm(self.conv3(x, stochastic=stochastic)))
        return x

