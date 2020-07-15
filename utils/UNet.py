import torch
from torch import nn

def conv2d_block(input_channels, out_channels=128, kernel_size = 3, batchnorm = False):
    """Function to add 2 convolutional layers with the parameters passed to it""" 
    p = 1 ##padding
    layers = []
    
    layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, padding=p))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels)) 
    layers.append(nn.ReLU(inplace=True))
    
    first_layer = nn.Sequential(*layers)
    
    # second layer
    layers = []
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=p))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels)) 
    layers.append(nn.ReLU(inplace=True))
    second_layer = nn.Sequential(*layers)
    
    # third layer
    layers = []
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=p))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels)) 
    #layers.append(nn.ReLU(inplace=True))
    third_layer = nn.Sequential(*layers)
    
    #return nn.Sequential(*layers)
    return nn.ModuleList([first_layer, second_layer, third_layer, nn.ReLU(inplace=True)])


  
class UNet(nn.Module):
    def __init__(self, input_channels=1, n_filters = 128, dropout = 0.1, batchnorm = False):
        super(UNet, self).__init__()
        
        
        # Contracting Path
        self.c1 = conv2d_block(input_channels, out_channels=n_filters, batchnorm = batchnorm)
        self.p1 = nn.AvgPool2d(2)

        self.c2 = conv2d_block(n_filters, out_channels=n_filters, batchnorm = batchnorm)
        self.p2 = nn.AvgPool2d(2)

        self.c3 = conv2d_block(n_filters, out_channels=n_filters, batchnorm = batchnorm)
        self.p3 = nn.AvgPool2d(2)


        self.c4 = conv2d_block(n_filters, out_channels=n_filters, batchnorm = batchnorm)
        self.p4 = nn.AvgPool2d(2)
        self.d4 = nn.Dropout(p=dropout, inplace=True)

        self.c5 = conv2d_block(n_filters, out_channels=n_filters, batchnorm = batchnorm)
        self.p5 = nn.AvgPool2d(2)
        self.d5 = nn.Dropout(p=dropout, inplace=True)

        self.feature_space = conv2d_block(n_filters, out_channels=n_filters, batchnorm = batchnorm)

        # Expansive Path
        self.upsample5 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, stride=2, padding=0) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.du5 = nn.Dropout(p=dropout, inplace=True)
        self.up5 = conv2d_block(n_filters*2, out_channels=n_filters,  batchnorm = batchnorm)

        self.upsample4 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, stride=2, padding=0) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.du4 = nn.Dropout(p=dropout, inplace=True)
        self.up4 = conv2d_block(n_filters*2, out_channels=n_filters, batchnorm = batchnorm)

        self.upsample3 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, stride=2, padding=0)  #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = conv2d_block(n_filters*2, out_channels=n_filters, batchnorm = batchnorm)

        self.upsample2 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, stride=2, padding=0) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = conv2d_block(n_filters*2, out_channels=n_filters,  batchnorm = batchnorm)

        self.upsample1 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, stride=2, padding=0) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = conv2d_block(n_filters*2, out_channels=n_filters,  batchnorm = batchnorm)
        
        self.last_conv = nn.Conv2d(n_filters, 1, kernel_size=1)
        
        #self.last_activation = nn.Sigmoid()
        
        self.initialize_weights()
    
    def pass_through_conv2d_block(self, x, conv_block, apply_skip=True):
        first_output = conv_block[0](x)
        x = conv_block[1](first_output)
        x = conv_block[2](x)
        if apply_skip is True:
            x = x + first_output
        x = conv_block[3](x)
        return x
        
    def forward(self, x):
       
        c1 = self.pass_through_conv2d_block(x, self.c1, apply_skip=False)
        x = self.p1(c1)
        
        c2 = self.pass_through_conv2d_block(x, self.c2)
        x = self.p2(c2)
        
        c3 = self.pass_through_conv2d_block(x, self.c3)
        x = self.p3(c3)
        
        c4 = self.pass_through_conv2d_block(x, self.c4)
        x = self.p4(c4)
        x = self.d4(x)
        
        c5 = self.pass_through_conv2d_block(x, self.c5)
        x = self.p5(c5)
        x = self.d5(x)
        
        x = self.pass_through_conv2d_block(x, self.feature_space)
        
        x = self.upsample5(x)
        x = torch.cat([x, c5], dim=1)
        x = self.du5(x)
        x = self.pass_through_conv2d_block(x, self.up5)
        
        x = self.upsample4(x)
        x = torch.cat([x, c4], dim=1)
        x = self.du4(x)
        x = self.pass_through_conv2d_block(x, self.up4)
        
        x = self.upsample3(x)
        x = torch.cat([x, c3], dim=1)
        x = self.pass_through_conv2d_block(x, self.up3)
        
        x = self.upsample2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.pass_through_conv2d_block(x, self.up2)
        
        x = self.upsample1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.pass_through_conv2d_block(x, self.up1)
        
        x = self.last_conv(x)
        
        return x
    def initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                classname = m.__class__.__name__
                #print(m)
                if classname.find('Conv2d') != -1:
                    torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.fill_(0)
            #print("last conv seperate")
            torch.nn.init.normal_(self.last_conv.weight.data, std=0.001)
            self.last_conv.bias.data.fill_(0)