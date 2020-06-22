import torch
import torchvision
from torch import nn
class Discriminator(nn.Module):
  #Dicriminator部分
  def __init__(self, nch=3, nch_d=16):
     super(Discriminator, self).__init__()
     self.layer1 = self.conv_layer(nch, nch_d, 3, 2, 1,False)
     self.layer2 = self.conv_layer(nch_d, nch_d * 2, 3, 2, 1,False)
     self.layer3 = self.conv_layer(nch_d * 2, nch_d * 4,3, 2, 1,False)
     self.layer4 = self.conv_layer(nch_d * 4, nch_d * 8, 3, 2, 1,False)
     self.layer5 = self.conv_layer(nch_d * 8, nch_d * 16, 3, 2, 1,False)
     self.layer6 = self.conv_layer(nch_d * 16, nch_d * 32, 3, 2, 1,False)
     self.layer7 = self.conv_layer(nch_d * 32, 1, 4, 1, 1,True)
  def multi_layers(self,layer1,layer2,layer3,z):
      z1 = self.convolution(layer1,z)
      z2 = self.convolution(layer2,z)
      z3 = self.convolution(layer2,z)
      z = z1+z2+z3
      z_copy = z

      return z

  def conv_layer(self,input,out,kernel_size,stride,padding,is_last):
      if is_last == True:
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.Conv2d(input , out , kernel_size, stride, padding,bias = False),
                  ),
              })
      else :
        return nn.ModuleDict({
               'layer0': nn.Sequential(
                  nn.Conv2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.BatchNorm2d(out),
                  nn.ReLU(),  
                  ), 
              })
        
  def convolution(self,layer_i,z):
      for layer in layer_i.values(): 
            z = layer(z)
      return z
  def forward(self, x):
      x = self.convolution(self.layer1,x)
      x = self.convolution(self.layer2,x)
      x = self.convolution(self.layer3,x)
      x = self.convolution(self.layer4,x)
      x = self.convolution(self.layer5,x)
      x = self.convolution(self.layer6,x)
      x = self.convolution(self.layer7,x)
      return x