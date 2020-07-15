import torch
from torch import nn

def get_spatial_conv(input_channels=33, num_filters=128, kernel_size=11, padding=5, count=3):
    layers = [nn.Conv2d(input_channels, num_filters, kernel_size, padding=padding), nn.LeakyReLU(0.1, inplace=True)]
    
    for c in range(count-2):
        layers.append(nn.Conv2d(num_filters, num_filters, kernel_size, padding=padding))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    
    #layers.append(nn.Conv2d(num_filters, input_channels, 11, padding=padding))
    #layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)

class SCN(nn.Module):
    def __init__(self, output_channels=33, num_filters=128):
        super(SCN, self).__init__()
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        ##local common
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.AvgPool = nn.AvgPool2d(2)
        #self.droput = nn.Dropout(p=0.5)
         
        ## local appearance net
        self.l_conv_1_1 = nn.Conv2d(1, num_filters, 3, padding=1)
        self.l_conv_1_2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_1_3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_2_1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_2_2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_2_3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_3_1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_3_2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_3_3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_4_1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_4_2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_conv_4_3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.l_final = nn.Conv2d(num_filters*4, self.output_channels, 3, padding=1)
        
        ##spatial common
        self.AvgPool16 = nn.AvgPool2d(16)
        self.upsample16 = nn.Upsample(scale_factor=16, mode="bicubic", align_corners=True)
        ##spatial apperance net
        self.spatial_conv = get_spatial_conv(output_channels, num_filters=num_filters)
        self.s_final = nn.Conv2d(num_filters, output_channels, 11, padding=5)
        self.tanh = nn.Tanh()
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            print("Initializing weights")
            torch.nn.init.kaiming_normal_(self.l_conv_1_1.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_1_1.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_1_2.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_1_2.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_1_3.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_1_3.bias.data.fill_(0)
            
            torch.nn.init.kaiming_normal_(self.l_conv_2_1.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_2_1.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_2_2.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_2_2.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_2_3.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_2_3.bias.data.fill_(0)
            
            torch.nn.init.kaiming_normal_(self.l_conv_3_1.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_3_1.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_3_2.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_3_2.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_3_3.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_3_3.bias.data.fill_(0)
            
            torch.nn.init.kaiming_normal_(self.l_conv_4_1.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_4_1.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_4_2.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_4_2.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.l_conv_4_3.weight.data, mode='fan_in', nonlinearity='relu')
            self.l_conv_4_3.bias.data.fill_(0)
            
            torch.nn.init.normal_(self.l_final.weight.data, std=0.001)
            self.l_final.bias.data.fill_(0)
            torch.nn.init.normal_(self.s_final.weight.data, std=0.001)
            self.s_final.bias.data.fill_(0)
            
            for m in self.spatial_conv:
                if m.__class__.__name__ == "Conv2d":
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                    m.bias.data.fill_(0)
           
    def forward(self, x):
        x = self.relu(self.l_conv_1_1(x))
        x = self.relu(self.l_conv_1_2(x))
        l_conv_1 = self.relu(self.l_conv_1_3(x))
        #print(l_conv_1.shape)
        
        x = self.AvgPool(x)
        
        x = self.relu(self.l_conv_2_1(x))
        x = self.relu(self.l_conv_2_2(x))
        l_conv_2 = self.relu(self.l_conv_2_3(x))
        #print(l_conv_2.shape)
        
        
        x = self.AvgPool(x)
        #x = self.droput(x)
        
        x = self.relu(self.l_conv_3_1(x))
        x = self.relu(self.l_conv_3_2(x))
        l_conv_3 = self.relu(self.l_conv_3_3(x))
        #print(l_conv_3.shape)
        
        
        x = self.AvgPool(x)
        #x = self.droput(x)
        
        x = self.relu(self.l_conv_4_1(x))
        x = self.relu(self.l_conv_4_2(x))
        x = self.relu(self.l_conv_4_3(x))
        
        #print("before starting upsample ", x.shape)
        
        x = self.upsample(x)
        x = torch.cat([x, l_conv_3], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, l_conv_2], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, l_conv_1], dim=1)
        
        hla = self.l_final(x)
        
        x = self.AvgPool16(hla)
        x = self.spatial_conv(x)
        x = self.tanh(self.s_final(x))
        hsc = self.upsample16(x)
        return hla*hsc