import torch
import torch.nn as nn
input_layouts=["CHW"]
ifmap_sz=[[8,32,32]]
op_version=14
batch_size=1
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift8=nn.Conv2d(8,8,kernel_size=3,padding=1,bias=False)
        coords=[(2,2),(2,1),(2,0),(1,2),(1,0),(0,2),(0,1),(0,0)]
        w=torch.zeros(8,8,3,3)
        for k,(r,c) in enumerate(coords):
            w[k,k,r,c]=1.0
        with torch.no_grad():
            self.shift8.weight.copy_(w)
        self.shift8.weight.requires_grad_(False)
    def forward(self,x):
        return self.shift8(x)
