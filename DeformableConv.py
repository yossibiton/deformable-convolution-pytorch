import torch
from torch import nn
from torch import autograd
from modules import ConvOffset2d


class DeformableConv(nn.Module):
    def __init__(self, inC, outC, kernel_size):
        super(DeformableConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=inC,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=(1, 1),
            bias=False).cuda()

        self.conv_offset2d = ConvOffset2d(
            in_channels=inC,
            out_channels=outC,
            kernel_size=kernel_size,
            padding=1
        ).cuda()

    def forward(self, x):
        offset = self.conv(x)
        output = self.conv_offset2d(x, offset)
        return output


if __name__ == '__main__':
    net = DeformableConv(inC=6, outC=4, kernel_size=(3,3))
    inputs = autograd.Variable(torch.randn(1, 6, 512, 512).cuda())
    output = net(inputs)
    output.backward(output.data)
    print(output.size())
