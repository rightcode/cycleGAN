import torch
import torchvision
from torch import nn
class Generator(nn.Module):
    def __init__(self,nch,nch_d):
        super(Generator, self).__init__()
        nch_g = 64
        #U-net部分
        self.layer1 = self.conv_layer_forward(nch, nch_g , 3, 2, 1,False)
        self.Res1 = self.Residual_module(nch_g)
        self.layer2 = self.conv_layer_forward(nch_g , nch_g*2 , 3, 2, 1,False)
        self.Res2 = self.Residual_module(nch_g*2)
        self.layer3 = self.conv_layer_forward(nch_g*2 , nch_g*4 , 3, 2, 1,False)
        self.Res3 = self.Residual_module(nch_g*4)
        self.layer4= self.conv_layer_forward(nch_g*4 , nch_g*8 , 3, 2, 1,False)
        self.Res4 = self.Residual_module(nch_g*8)
        self.layer5= self.conv_layer_forward(nch_g*8 , nch_g*16 , 3, 2, 1,False)
        self.Res5 = self.Residual_module(nch_g*16)
        self.layer6= self.conv_layer_forward(nch_g*16 , nch_g*32 , 3, 2, 1,False)
        self.layer7= self.conv_layer_transpose(nch_g*32 , nch_g*16 , 4, 2, 1,False)
        self.Res7 = self.Residual_module(nch_g*32)
        self.layer8 = self.conv_layer_transpose(nch_g*32 , nch_g*8 , 4, 2, 1,False)
        self.Res8 = self.Residual_module(nch_g*16)
        self.layer9 = self.conv_layer_transpose(nch_g*16 , nch_g*4 , 4, 2, 1,False)
        self.Res9 = self.Residual_module(nch_g*8)
        self.layer10= self.conv_layer_transpose(nch_g*8 , nch_g*2 , 4, 2, 1,False)
        self.Res10 = self.Residual_module(nch_g*4)
        self.layer11= self.conv_layer_transpose(nch_g*4 , nch_g , 4, 2, 1,False)
        self.Res11 = self.Residual_module(nch_g*2)
        self.layer12 = self.conv_layer_transpose(nch_g*2 , nch_g , 4, 2, 1,False)
        self.Res12 = self.Residual_module(nch_g)
        self.layer13= self.conv_layer_forward(nch_g , nch_d , 1, 1, 0,True)

    def forward(self, z):
        z,z1 = self.convolution_forward(self.layer1,z)
        z,z1 = self.ResNet(self.Res1,z,z1)
        
        z,z2= self.convolution_forward(self.layer2,z)
        z,z2 = self.ResNet(self.Res2,z,z2)

        z,z3 = self.convolution_forward(self.layer3,z)
        z,z3 = self.ResNet(self.Res3,z,z3)

        z,z4 = self.convolution_forward(self.layer4,z)
        z,z4 = self.ResNet(self.Res4,z,z4)

        z,z5 = self.convolution_forward(self.layer5,z)
        z,z5 = self.ResNet(self.Res5,z,z5)

        z,_ = self.convolution(self.layer6,z)

        z,z_copy = self.convolution_deconv(self.layer7,z,z5)
        z,_ = self.ResNet(self.Res7,z,z_copy)

        z,z_copy = self.convolution_deconv(self.layer8,z,z4)
        z,_ = self.ResNet(self.Res8,z,z_copy)

        z,z_copy = self.convolution_deconv(self.layer9,z,z3)
        z,_ = self.ResNet(self.Res9,z,z_copy)

        z,z_copy = self.convolution_deconv(self.layer10,z,z2)
        z,_ = self.ResNet(self.Res10,z,z_copy)

        z,z_copy = self.convolution_deconv(self.layer11,z,z1)
        z,_ = self.ResNet(self.Res11,z,z_copy)

        z,z_copy = self.convolution(self.layer12,z)
        z,_ = self.ResNet(self.Res12,z,z_copy)

        z,_ = self.convolution(self.layer13,z)
        return z

    def convolution(self,layer_i,z):
      for layer in layer_i.values(): 
            z = layer(z)
      z_copy = z
      return z,z_copy
    def ResNet(self,layer,z,pre_z):
        z, _ = self.convolution(layer,z)
        z = z+pre_z
        z_copy = z
        return z,z_copy

    def Residual_module(self,input):
      return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,int(input/2),3,1,1,bias = False),
                    nn.BatchNorm2d(int(input/2)),
                    nn.ReLU(),  
                    ),  
                'layer1': nn.Sequential(
                    nn.Conv2d(int(input/2),int(input/2),1,1,0,bias = False),
                    nn.BatchNorm2d(int(input/2)),
                    nn.ReLU(),  
                    ),  
                'layer2': nn.Sequential(
                    nn.Conv2d(int(input/2),input,3,1,1,bias = False),
                    nn.BatchNorm2d(input),
                    nn.ReLU(),  
                    ),  
                })
    
    def conv_layer_forward(self,input,out,kernel_size,stride,padding,is_last):
        if is_last == False:
          return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                    nn.BatchNorm2d(out),
                    nn.ReLU(),  
                    ),  
                })
        else:
          return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                    nn.Tanh()
                    ),  
                })
        
    def conv_layer_forward_image_size_1(self,input,out,kernel_size,stride,padding):
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                  nn.ReLU(),  
                  ),  
              })
        
    def conv_layer_transpose(self,input,out,kernel_size,stride,padding,is_last):
      if is_last == True:
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.ConvTranspose2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.Tanh()  
                  ),
              })
      else :
        return nn.ModuleDict({
               'layer0': nn.Sequential(
                  nn.ConvTranspose2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.BatchNorm2d(out),
                  nn.ReLU(),  
                  ), 
              })
        
    def convolution_forward(self,layer,z):
        z,z_copy = self.convolution(layer,z)
        return z,z_copy
    def convolution_deconv(self,layer,z,z_copy):
        z,_ = self.convolution(layer,z)
        z = torch.cat([z,z_copy],dim = 1)
        z_copy = z
        return z,z_copy
