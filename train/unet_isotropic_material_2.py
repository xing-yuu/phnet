import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_in*2, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_in*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in*2, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class pure_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(pure_conv_block,self).__init__()
        self.pure_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
        )


    def forward(self,x):
        x = self.pure_conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,int(ch_in/2),kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(int(ch_in/2)),
			nn.ReLU(inplace=True),
            nn.Conv3d(int(ch_in/2),ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv_0(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_0,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=(9,9,9)),
            nn.Conv3d(ch_in,int(ch_in/2),kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(int(ch_in/2)),
			nn.ReLU(inplace=True),
            nn.Conv3d(int(ch_in/2),ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x




class U_Net(nn.Module):
    def __init__(self,init_ch=2,output_ch=18):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        # 36 -> 18
        self.Conv1 = conv_block(ch_in=init_ch,ch_out=8)
        # 18 -> 9
        self.Conv2 = conv_block(ch_in=8,ch_out=32)
        # # 9 -> 5
        self.Conv3 = conv_block(ch_in=32,ch_out=128)
        #  # # 5 -> 3
        self.Conv4 = conv_block(ch_in=128,ch_out=512)

        self.Up4 = up_conv_0(ch_in=512,ch_out=128)

        self.Up3 = up_conv(ch_in=256,ch_out=64)

        self.Up2 = up_conv(ch_in=96,ch_out=24)
        self.pure_conv = pure_conv_block(ch_in=32, ch_out=32)

        self.Conv_1x1 = nn.Conv3d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # print(x.shape)
        # encoding path
        x1 = self.Conv1(x) # 8*36^3

        x2 = self.Maxpool(x1) #8*18^3
        x2 = self.Conv2(x2)   #32*18^3
        
        x3 = self.Maxpool(x2) #32*9^3
        x3 = self.Conv3(x3)   #128*9^3
    
        x4 = self.Maxpool(x3) #128*5^3
        x4 = self.Conv4(x4)   #512*5^3

        d4=self.Up4(x4)#128*9^3
        d3=torch.cat((x3,d4),dim=1) #256*9^3

        d3 = self.Up3(d3) #64*18^3
        d3 = torch.cat((x2,d3),dim=1) #96*18^3

        d2 = self.Up2(d3) # 24*36^3
        d2 = torch.cat((x1,d2),dim=1) #32*36^3
        d2 = self.pure_conv(d2) #32*36^3

        d1 = self.Conv_1x1(d2) ##18*36^3

        return d1


