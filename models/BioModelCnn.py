import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class BioModelCnn(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(BioModelCnn, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128]
        self.conv1 = downsample_conv(1,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])

        upconv_planes = [128, 64, 32, 16]
        self.upconv4 = upconv(conv_planes[2], upconv_planes[0])
        self.upconv3 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv2 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv1 = upconv(upconv_planes[2], upconv_planes[3])

        fc_nodes = [64, 32, 10]

        self.fc1 = nn.Linear(16 * 400 * 16, fc_nodes[0])
        self.fc2 = nn.Linear(fc_nodes[0], fc_nodes[1])
        self.fc3 = nn.Linear(fc_nodes[1], fc_nodes[2])

        self.bio_pred = nn.Linear(fc_nodes[2], 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_upconv4 = self.upconv4(out_conv3)
        out_upconv5 = self.upconv3(out_upconv4)
        out_upconv6 = self.upconv2(out_upconv5)
        out_upconv7 = self.upconv1(out_upconv6)

        # print("upconvplanes: ", out_upconv7.size())
        out = out_upconv7.view(-1, 16 * 400 * 16)
        # print("out: ", out.size())

        out_fc1 = self.fc1(out)
        # print("out_fc1: ", out_fc1.size())

        out_fc2 = self.fc2(out_fc1)
        # print("out_fc2: ", out_fc2.size())

        out_fc3 = self.fc3(out_fc2)
        # print("out_fc3: ", out_fc3.size())

        bio_pred = self.bio_pred(out_fc3)

        # print("bio_pred: ", bio_pred.size())

        return bio_pred
