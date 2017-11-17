import torch
import torch.nn as nn
from torch.autograd import Variable

from modules import ConvOffset2d

N, inC, inH, inW = 1, 6, 512, 512
outC, outH, outW = 4, 512, 512
kH, kW = 3, 3

conv = nn.Conv2d(
    in_channels=inC,
    out_channels=2 * kH * kW,
    kernel_size=(kH, kW),
    padding=(1, 1),
    bias=False).cuda()

conv_offset2d = ConvOffset2d(
    in_channels=inC,
    out_channels=outC, 
    kernel_size=(kH, kW),
    padding=1).cuda()

inputs = Variable(torch.randn(N, inC, inH, inW).cuda())
offset = conv(inputs)
output = conv_offset2d(inputs, offset)
output.backward(output.data)
print(output.size())
