from torch import nn
import torch

class FeatureBlock(nn.Module):
    def __init__(self, n_channels=128, apply_bn=False):
        super(FeatureBlock, self).__init__()
        kernel_size = 3
        padding = 1
        self.apply_bn = apply_bn
        self.conv1 = nn.Conv2d(n_channels, n_channels, padding=padding, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(n_channels, n_channels, padding=padding, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(n_channels, n_channels, padding=padding, kernel_size=kernel_size)
        if apply_bn:
            self.bns = nn.ModuleList([nn.BatchNorm2d(n_channels)]*3)
        self.activates = nn.ModuleList([nn.ReLU(inplace=False)]*3)
    
    def forward(self, x):
        og_x = x
        x = self.conv1(x)
        if self.apply_bn:
            x = self.bns[0](x)
        x = self.activates[0](x)
        
        x = self.conv2(x)
        if self.apply_bn:
            x = self.bns[1](x)
        x = self.activates[1](x)
        
        x = self.conv3(x)
        x = x + og_x
        if self.apply_bn:
            x = self.bns[2](x)
        x = self.activates[2](x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_channels=128, n_filters=128, apply_bn = False, apply_res=True):
        super(EncoderBlock, self).__init__()
        self.input_channels = input_channels
        self.apply_bn = apply_bn
        self.apply_res = apply_res
        kernel_size=3
        p=1
        
        self.first_conv = nn.Conv2d(input_channels, n_filters, kernel_size, padding=p)
        self.first_activate = nn.ReLU(inplace=False)
        if apply_bn is True:
            self.first_bn = nn.BatchNorm2d(n_filters)
        
        self.second_conv = nn.Conv2d(n_filters, n_filters, kernel_size, padding=p)
        self.second_activate = nn.ReLU(inplace=False)
        if apply_bn is True:
            self.second_bn = nn.BatchNorm2d(n_filters)
        self.AvgPool = nn.AvgPool2d(2)
    
    def forward(self, x):
        first_op = x
        
        x = self.first_conv(x)
        if self.apply_bn:
            x = self.first_bn(x)
        x = self.first_activate(x)
        
        x = self.second_conv(x)
       
        if self.apply_res:
            x  = x + first_op
        if self.apply_bn:
            x = self.second_bn(x)
        x = self.second_activate(x)
        skip = x
        x = self.AvgPool(x)
        return x, skip

class DeconvDecoderBlock(nn.Module):
    def __init__(self, in_channels=128*2,  n_filters=128, apply_bn=True, apply_conv=False):
        super(DeconvDecoderBlock, self).__init__()
        self.apply_bn = apply_bn
        self.apply_conv = apply_conv
        kernel_size=2
        stride = 2
        p=0
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding=p)
        if apply_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=False),
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=False)
            )
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        if self.apply_conv:
            x = self.conv(x)
        return x
        

class DecoderBlock(nn.Module):
    def __init__(self, n_filters=128, apply_bn = False, dp=False):
        super(DecoderBlock, self).__init__()
        self.apply_bn = apply_bn
        self.dp = dp
        kernel_size=3 
        p=1
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.first_conv = nn.Conv2d(n_filters*2, n_filters, kernel_size, padding=p)
        if apply_bn:
            self.first_bn = nn.BatchNorm2d(n_filters)
        self.first_activate = nn.ReLU(inplace=False)
        self.second_conv = nn.Conv2d(n_filters, n_filters, kernel_size, padding=p)
        if apply_bn:
            self.second_bn = nn.BatchNorm2d(n_filters)
        self.second_activate = nn.ReLU(inplace=False)
        if dp is not False:
            self.dp = nn.Dropout(p=dp, inplace=False)
    def forward(self, x, skip):
        x = self.upsample(x)
        x  = torch.cat((x, skip), dim=1)
        
        if self.dp is not False:
            x = self.dp(x)
        
        x = self.first_conv(x)
        if self.apply_bn:
            x = self.first_bn(x)
        x = self.first_activate(x)
        
        
        x = self.second_conv(x)
        if self.apply_bn:
            x = self.second_bn(x)
        x = self.second_activate(x)
        
        return x

class DilatedBlock(nn.Module):
    def __init__(self, n_channels=128, dilation_rates=[1, 2, 4, 8, 16, 32], kernel_size=3, padding=1, apply_bn=False, dp=False):
        super(DilatedBlock, self).__init__()
        self.apply_bn = apply_bn
        self.dp = dp
        self.conv1 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[0], kernel_size=kernel_size, dilation=dilation_rates[0])
        self.conv2 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[1], kernel_size=kernel_size, dilation=dilation_rates[1])
        self.conv3 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[2], kernel_size=kernel_size, dilation=dilation_rates[2])
        self.conv4 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[3], kernel_size=kernel_size, dilation=dilation_rates[3])
        self.conv5 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[4], kernel_size=kernel_size, dilation=dilation_rates[4])
        self.conv6 = nn.Conv2d(n_channels, n_channels, padding=dilation_rates[5], kernel_size=kernel_size, dilation=dilation_rates[5])
        if apply_bn:
            self.bns = nn.ModuleList([nn.BatchNorm2d(n_channels)]*6)
        self.activates = nn.ModuleList([nn.ReLU(inplace=False)]*6)
        if dp is not False:
            self.dps = nn.ModuleList([nn.Dropout(p=dp, inplace=False)]*6)
    def forward(self, x):
       
        x = self.conv1(x)
        if self.apply_bn :
            x = self.bns[0](x)
        x = self.activates[0](x)
        dilate1 = x
        if self.dp is not False:
            x = self.dps[0](x)
            
        x = self.conv2(x)
        if self.apply_bn :
            x = self.bns[1](x)
        x = self.activates[1](x)
        dilate2 = x
        if self.dp is not False:
            x = self.dps[1](x)
        
        x = self.conv3(x)
        if self.apply_bn :
            x = self.bns[2](x)
        x = self.activates[2](x)
        dilate3 = x
        if self.dp is not False:
            x = self.dps[2](x)
        
        x = self.conv4(x)
        if self.apply_bn :
            x = self.bns[3](x)
        x = self.activates[3](x)
        dilate4 = x
        if self.dp is not False:
            x = self.dps[3](x)
        
        x = self.conv5(x)
        if self.apply_bn :
            x = self.bns[4](x)
        x = self.activates[4](x)
        dilate5 = x
        if self.dp is not False:
            x = self.dps[4](x)
            
        x = self.conv6(x)
        if self.apply_bn :
            x = self.bns[5](x)
        x = self.activates[5](x)
        dilate6 = x
        
        
        dilate = dilate1 + dilate2 + dilate3 + dilate4 + dilate5 + dilate6
        
        return dilate
class UNetDR(nn.Module):
    def __init__(self, n_channels=128, apply_bn=True, dropout=0.1):
        super(UNetDR, self).__init__()
        self.n_channels = n_channels
        self.EB1 = EncoderBlock(1, n_channels, apply_bn=apply_bn, apply_res=False)
        #self.AvgPools = nn.ModuleList([nn.AvgPool2d(2)]*5)
        self.EB2 = EncoderBlock(n_channels, n_channels, apply_bn=apply_bn)
        self.EB3 = EncoderBlock(n_channels, n_channels, apply_bn=apply_bn)
        self.EB4 = EncoderBlock(n_channels, n_channels, apply_bn=apply_bn)
        self.dp4 = nn.Dropout(p=dropout, inplace=False)
        self.EB5 = EncoderBlock(n_channels, n_channels, apply_bn=apply_bn)
        self.dp5 = nn.Dropout(p=dropout, inplace=False)
        self.DB5 = DecoderBlock(n_channels, apply_bn=apply_bn, dp=dropout)
        self.DB4 = DecoderBlock(n_channels, apply_bn=apply_bn, dp=dropout)
        self.DB3 = DecoderBlock(n_channels, apply_bn=apply_bn, dp=False)
        self.DB2 = DecoderBlock(n_channels, apply_bn=apply_bn, dp=False)
        self.DB1 = DecoderBlock(n_channels, apply_bn=apply_bn, dp=False)
        
        self.last_conv = nn.Conv2d(n_channels, 1, 3, padding=1)
        self.DB = DilatedBlock(n_channels, apply_bn=apply_bn, dp=False)
        
        self.initialize_weights()
    def forward(self, x):
        x, skip1 = self.EB1(x)
        x, skip2 = self.EB2(x)
        x, skip3 = self.EB3(x)
        x, skip4 = self.EB4(x)
        x = self.dp4(x)
        x, skip5 = self.EB5(x)
        x = self.dp5(x)
        
        x = self.DB(x)
        
        x = self.DB5(x, skip5)
        x = self.DB4(x, skip4)
        x = self.DB3(x, skip3)
        x = self.DB2(x, skip2)
        x = self.DB1(x, skip1)
       
        x = self.last_conv(x)
        return x
    def initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                classname = m.__class__.__name__
                if classname.find('Conv2d') != -1:
                    torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.fill_(0)
            torch.nn.init.normal_(self.last_conv.weight.data, std=0.001)
            self.last_conv.bias.data.fill_(0)
        