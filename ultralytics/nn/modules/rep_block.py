import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.conv import Conv, autopad

__all__ = ['DiverseBranchBlock','DeepACBlockDBB','WideDiverseBranchBlock','DeepDiverseBranchBlock','ACBlockDBB','ACBlock']

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Module):
    def __init__(self, channels, groups=1):
        super().__init__()
        assert channels % groups == 0
        input_dim = channels // groups
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups, bias=False)
        
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value)
        nn.init.zeros_(self.conv.weight)
        self.groups = groups
    
    def forward(self, input):
        kernel = self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)
        result = F.conv2d(input, kernel, None, stride=1, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)

class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))


            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DiverseBranchBlockNOAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlockNOAct, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            # padding=None
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)

        # print(inputs.shape)
        # print(self.dbb_1x1(inputs).shape)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    @property
    def weight(self):  ##含有@property
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam.weight

class DeepDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlock, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        # padding=0
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))
            # return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

        # return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DeepDiverseBranchBlockNOAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlockNOAct, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        # padding=0
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class WideDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(WideDiverseBranchBlock, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

        if padding - kernel_size // 2 >= 0:
            self.crop = 0
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:
            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]

            # Vertical convolution(3x1) during training
        self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=ver_padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=False,
                                    )
        # Horizontal convolution(1x3) during training
        self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=hor_padding,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=False,
                                  )
        # Batch normalization for vertical convolution
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
                                     affine=True)
        # Batch normalization for horizontal convolution
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
                                     affine=True)

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''
        Used to add an asymmetric kernel to the center of a square kernel
        square_kernel : the square kernel to which the asymmetric kernel will be added
        asym_kernel   : the asymmetric kernel that will be added to the square kernel
        '''
        # Get the height and width of the asymmetric kernel
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        # Get the height and width of the square kernel
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        # Add the asymmetric kernel to the center of the square kernel
        square_kernel[:,
        :,
        square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias_1xk_kx1_kxk(self):
        '''
        Used to calculate the equivalent kernel and bias of
        the fused convolution layer in deploy mode
        '''
        # Fuse batch normalization with convolutional weights and biases
        hor_k, hor_b       = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        ver_k, ver_b       = transI_fusebn(self.ver_conv.weight, self.ver_bn)
        square_k, square_b = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)


        # Add the fused horizontal and vertical kernels to the center of the square kernel
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        # Return the square kernel and the sum of the biases for the three convolutional layers
        return square_k, hor_b + ver_b + square_b


    def get_equivalent_kernel_bias(self):
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)
        k_origin, b_origin = self.get_equivalent_kernel_bias_1xk_kx1_kxk()


        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')


    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)

        if self.crop > 0:
            ver_input = inputs[:, :, :, self.crop:-self.crop]
            hor_input = inputs[:, :, self.crop:-self.crop, :]
        else:
            ver_input = inputs
            hor_input = inputs
        vertical_outputs = self.ver_conv(ver_input)
        vertical_outputs = self.ver_bn(vertical_outputs)
        horizontal_outputs = self.hor_conv(hor_input)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        result = out + vertical_outputs + horizontal_outputs

        return self.nonlinear(result)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DiverseBranchBlockNOAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlockNOAct, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            # padding=None
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)

        # print(inputs.shape)
        # print(self.dbb_1x1(inputs).shape)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    @property
    def weight(self):  ##含有@property
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam.weight

class DiverseBranchBlockNOActNopad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlockNOActNopad, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        padding=0
        # if padding is None:
        #     # padding=None
        #     padding = autopad(kernel_size, padding, dilation)
        # assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=0, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)

        # print(inputs.shape)
        # print(self.dbb_1x1(inputs).shape)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    @property
    def weight(self):  ##含有@property
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam.weight


class DeepDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlock, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        # padding=0
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))
            # return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

        # return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DeepDiverseBranchBlockNOAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlockNOAct, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        # padding=0
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DeepDiverseBranchBlockNOAct3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlockNOAct3, self).__init__()
        self.deploy = deploy

        # self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        # padding=0
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        DiverseBranchBlockNOActNopad(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  ))
            # self.dbb_1x1_kxk.add_module('conv2',
            #                             nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
            #                                       kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
            #                                       bias=False))
            # self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        # k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_second, b_1x1_kxk_second = self.dbb_1x1_kxk.conv2.dbb_reparam.weight, self.dbb_1x1_kxk.conv2.dbb_reparam.bias
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        # return self.nonlinear(out)

        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


# class ACBlockDBB(nn.Module):
#     def __init__(self,
#                  in_channels, out_channels, kernel_size,
#                  stride=1, padding=None, dilation=1,
#                  groups=1, padding_mode='zeros',
#                  deploy=False, use_affine=True, n=3):
#         super(ACBlockDBB, self).__init__()
#         if padding is None:
#             padding = autopad(kernel_size, padding, dilation)
#         self.deploy = deploy
#         if deploy:
#             # # Fused convolution(3x3) during inference (re-parameterization)
#             self.fused_conv = nn.Conv2d(in_channels=in_channels,
#                                         out_channels=out_channels,
#                                         kernel_size=(kernel_size, kernel_size),
#                                         stride=stride,
#                                         padding=padding,
#                                         dilation=dilation,
#                                         groups=groups,
#                                         bias=True,
#                                         padding_mode=padding_mode)
#         else:
#             # Square convolution(3x3) during training
#             # print(padding)
#             self.square_conv = nn.Conv2d(in_channels=in_channels,
#                                          out_channels=out_channels,
#                                          kernel_size=(kernel_size, kernel_size),
#                                          stride=stride,
#                                          padding=padding,
#                                          dilation=dilation,
#                                          groups=groups,
#                                          bias=True,
#                                          padding_mode=padding_mode)
#             # 设置卷积核的requires_grad为False
#             self.square_conv.weight.requires_grad = False
#             self.square_conv.bias.requires_grad = False
#
#             self.mmlist_square_conv = torch.nn.ModuleList((DiverseBranchBlock(in_channels=in_channels,
#                                                                               out_channels=out_channels,
#                                                                               kernel_size=kernel_size,
#                                                                               stride=stride,
#                                                                               padding=padding,
#                                                                               dilation=dilation,
#                                                                               groups=groups,
#                                                                               ) for _ in range(n)))
#
#             # self.mmlist_square_bn=torch.nn.ModuleList(nn.BatchNorm2d(num_features=out_channels, affine=use_affine)  for _ in range(n))
#             self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
#
#             # Used to check whether the specified padding value is large enough
#             # to ensure that the convolution operation can be applied to
#             # all pixels in the input tensor without exceeding its boundaries
#             if padding - kernel_size // 2 >= 0:
#                 self.crop = 0
#                 hor_padding = [padding - kernel_size // 2, padding]
#                 ver_padding = [padding, padding - kernel_size // 2]
#             else:
#                 self.crop = kernel_size // 2 - padding
#                 hor_padding = [0, padding]
#                 ver_padding = [padding, 0]
#
#             # Vertical convolution(3x1) during training
#             self.ver_conv = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels,
#                                       kernel_size=(kernel_size, 1),
#                                       stride=stride,
#                                       padding=ver_padding,
#                                       dilation=dilation,
#                                       groups=groups,
#                                       bias=False,
#                                       padding_mode=padding_mode)
#             # Horizontal convolution(1x3) during training
#             self.hor_conv = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels,
#                                       kernel_size=(1, kernel_size),
#                                       stride=stride,
#                                       padding=hor_padding,
#                                       dilation=dilation,
#                                       groups=groups,
#                                       bias=False,
#                                       padding_mode=padding_mode)
#             # Batch normalization for vertical convolution
#             self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
#                                          affine=use_affine)
#             # Batch normalization for horizontal convolution
#             self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
#                                          affine=use_affine)
#
#     def forward(self, input):
#         if self.deploy:
#             return self.fused_conv(input)
#         else:
#             square_outputs_list = []
#             # for square_conv,square_bn in zip(self.mmlist_square_conv,self.mmlist_square_bn):
#             #     square_outputs_list.append(square_bn(square_conv(input)))
#             for square_conv in self.mmlist_square_conv:
#                 square_outputs_list.append(square_conv(input))
#
#             # 使用torch.stack将张量堆叠在一起
#             stacked_tensor = torch.stack(square_outputs_list)
#
#             # 使用torch.sum计算张量的和
#             square_outputs = torch.sum(stacked_tensor, dim=0)
#
#             square_outputs = self.square_bn(square_outputs)
#             if self.crop > 0:
#                 ver_input = input[:, :, :, self.crop:-self.crop]
#                 hor_input = input[:, :, self.crop:-self.crop, :]
#             else:
#                 ver_input = input
#                 hor_input = input
#             vertical_outputs = self.ver_conv(ver_input)
#             vertical_outputs = self.ver_bn(vertical_outputs)
#             horizontal_outputs = self.hor_conv(hor_input)
#             horizontal_outputs = self.hor_bn(horizontal_outputs)
#             result = square_outputs + vertical_outputs + horizontal_outputs
#             return result
#
#     def switch_to_deploy(self):
#
#         # Set deploy flag to True
#         self.deploy = True
#         # Get the fused kernel and bias
#
#         deploy_k, deploy_b = self.get_equivalent_kernel_bias()
#         # Define the fused convolutional layer
#
#         self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels,
#                                     out_channels=self.square_conv.out_channels,
#                                     kernel_size=self.square_conv.kernel_size,
#                                     stride=self.square_conv.stride,
#                                     padding=self.square_conv.padding,
#                                     dilation=self.square_conv.dilation,
#                                     groups=self.square_conv.groups,
#                                     bias=True,
#                                     padding_mode=self.square_conv.padding_mode)
#         # Delete the original convolutional and batch normalization layers
#         self.__delattr__('square_conv')
#         self.__delattr__('square_bn')
#         self.__delattr__('hor_conv')
#         self.__delattr__('hor_bn')
#         self.__delattr__('ver_conv')
#         self.__delattr__('ver_bn')
#         # self.__delattr__('square_conv_dbb_sum')
#         # Set the weight and bias of the fused convolutional layer
#         self.fused_conv.weight.data = deploy_k
#         self.fused_conv.bias.data = deploy_b
#
#     def get_equivalent_kernel_bias(self):
#         '''
#         Used to calculate the equivalent kernel and bias of
#         the fused convolution layer in deploy mode
#         '''
#         # Fuse batch normalization with convolutional weights and biases
#         hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
#         ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
#         self._fuse_dbb_conv()
#         square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
#
#         # Add the fused horizontal and vertical kernels to the center of the square kernel
#         self._add_to_square_kernel(square_k, hor_k)
#         self._add_to_square_kernel(square_k, ver_k)
#         # Return the square kernel and the sum of the biases for the three convolutional layers
#         return square_k, hor_b + ver_b + square_b
#
#     def _fuse_dbb_conv(self):
#
#         for dbb_conv in self.mmlist_square_conv:
#             dbb_conv.switch_to_deploy()
#             self.square_conv.weight += dbb_conv.dbb_reparam.weight
#             self.square_conv.bias += dbb_conv.dbb_reparam.bias
#
#     def _fuse_bn_tensor(self, conv, bn):
#         '''
#         Fusing Conv and BN
#         '''
#         # Calculate standard deviation
#         std = (bn.running_var + bn.eps).sqrt()
#         # Calculate the tensor for fusing batch normalization into convolution
#         t = (bn.weight / std).reshape(-1, 1, 1, 1)
#         # Return the fused weight and bias tensors
#         return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std
#
#     def _add_to_square_kernel(self, square_kernel, asym_kernel):
#         '''
#         Used to add an asymmetric kernel to the center of a square kernel
#         square_kernel : the square kernel to which the asymmetric kernel will be added
#         asym_kernel   : the asymmetric kernel that will be added to the square kernel
#         '''
#         # Get the height and width of the asymmetric kernel
#         asym_h = asym_kernel.size(2)
#         asym_w = asym_kernel.size(3)
#         # Get the height and width of the square kernel
#         square_h = square_kernel.size(2)
#         square_w = square_kernel.size(3)
#         # Add the asymmetric kernel to the center of the square kernel
#         square_kernel[:,
#         :,
#         square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
#         square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

class ACBlockDBB(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1,
                 groups=1, padding_mode='zeros',
                 deploy=False, use_affine=True, n=3):
        super(ACBlockDBB, self).__init__()
        self.deploy = deploy
        self.nonlinear = Conv.default_act
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        if deploy:
            # # Fused convolution(3x3) during inference (re-parameterization)
            self.fused_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=True,
                                        padding_mode=padding_mode)
        else:
            # Square convolution(3x3) during training
            # print(padding)
            self.square_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)
            # 设置卷积核的requires_grad为False
            self.square_conv.weight.requires_grad = False
            self.square_conv.bias.requires_grad = False

            self.mmlist_square_conv = torch.nn.ModuleList((DiverseBranchBlockNOAct(in_channels=in_channels,
                                                                                   out_channels=out_channels,
                                                                                   kernel_size=kernel_size,
                                                                                   stride=stride,
                                                                                   padding=padding,
                                                                                   dilation=dilation,
                                                                                   groups=groups,
                                                                                   ) for _ in range(n)))

            # self.mmlist_square_bn=torch.nn.ModuleList(nn.BatchNorm2d(num_features=out_channels, affine=use_affine)  for _ in range(n))
            # self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            # Used to check whether the specified padding value is large enough
            # to ensure that the convolution operation can be applied to
            # all pixels in the input tensor without exceeding its boundaries
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            # Vertical convolution(3x1) during training
            self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Horizontal convolution(1x3) during training
            self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Batch normalization for vertical convolution
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)
            # Batch normalization for horizontal convolution
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)

    def forward(self, input):
        if self.deploy:
            return self.nonlinear(self.fused_conv(input))
        else:
            square_outputs_list = []
            # for square_conv,square_bn in zip(self.mmlist_square_conv,self.mmlist_square_bn):
            #     square_outputs_list.append(square_bn(square_conv(input)))
            for square_conv in self.mmlist_square_conv:
                square_outputs_list.append(square_conv(input))

            # 使用torch.stack将张量堆叠在一起
            stacked_tensor = torch.stack(square_outputs_list)

            # 使用torch.sum计算张量的和
            square_outputs = torch.sum(stacked_tensor, dim=0)

            # square_outputs=self.square_bn(square_outputs)
            # square_outputs=self.nonlinear(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + (horizontal_outputs)
            return self.nonlinear(result)
            # return result

    def switch_to_deploy(self):

        # Set deploy flag to True
        self.deploy = True
        # Get the fused kernel and bias

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        # Define the fused convolutional layer

        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels,
                                    out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size,
                                    stride=self.square_conv.stride,
                                    padding=self.square_conv.padding,
                                    dilation=self.square_conv.dilation,
                                    groups=self.square_conv.groups,
                                    bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        # Delete the original convolutional and batch normalization layers
        self.__delattr__('square_conv')
        # self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.__delattr__('mmlist_square_conv')
        # self.__delattr__('square_conv_dbb_sum')
        # Set the weight and bias of the fused convolutional layer
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def get_equivalent_kernel_bias(self):
        '''
        Used to calculate the equivalent kernel and bias of
        the fused convolution layer in deploy mode
        '''
        # Fuse batch normalization with convolutional weights and biases
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        self._fuse_dbb_conv()
        # square_k, square_b =self._fuse_bn_tensor(self.square_conv, self.square_bn)
        square_k, square_b = self.square_conv.weight, self.square_conv.bias

        # Add the fused horizontal and vertical kernels to the center of the square kernel
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        # Return the square kernel and the sum of the biases for the three convolutional layers
        return square_k, hor_b + ver_b + square_b

    def _fuse_dbb_conv(self):
        self.square_conv.weight.data = torch.zeros_like(self.square_conv.weight.data)
        self.square_conv.bias.data = torch.zeros_like(self.square_conv.bias.data)
        for dbb_conv in self.mmlist_square_conv:
            dbb_conv.switch_to_deploy()
            self.square_conv.weight.data = torch.add(self.square_conv.weight.data, dbb_conv.dbb_reparam.weight.data)
            self.square_conv.bias.data = torch.add(self.square_conv.bias.data, dbb_conv.dbb_reparam.bias.data)
            # self.square_conv.weight.data = torch.add(self.square_conv.weight.data, dbb_conv.weight.data)
            # self.square_conv.bias.data = torch.add(self.square_conv.bias.data, dbb_conv.bias.data)
            # self.square_conv.weight.data  +=dbb_conv.dbb_reparam.weight.data
            # self.square_conv.bias.data  += dbb_conv.dbb_reparam.bias.data

    def _fuse_bn_tensor(self, conv, bn):
        '''
        Fusing Conv and BN
        '''
        # Calculate standard deviation
        std = (bn.running_var + bn.eps).sqrt()
        # Calculate the tensor for fusing batch normalization into convolution
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        # Return the fused weight and bias tensors
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''
        Used to add an asymmetric kernel to the center of a square kernel
        square_kernel : the square kernel to which the asymmetric kernel will be added
        asym_kernel   : the asymmetric kernel that will be added to the square kernel
        '''
        # Get the height and width of the asymmetric kernel
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        # Get the height and width of the square kernel
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        # Add the asymmetric kernel to the center of the square kernel
        square_kernel[:,
        :,
        square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

class FusedDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(FusedDiverseBranchBlock, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.dbb_1=DiverseBranchBlock(in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False)

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

class DeepACBlockDBB(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1,
                 groups=1, padding_mode='zeros',
                 deploy=False, use_affine=True, n=1):
        super(DeepACBlockDBB, self).__init__()
        self.deploy = deploy
        self.nonlinear = Conv.default_act
        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        if deploy:
            # # Fused convolution(3x3) during inference (re-parameterization)
            self.fused_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=True,
                                        padding_mode=padding_mode)
        else:
            # Square convolution(3x3) during training
            # print(padding)
            self.square_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)
            # 设置卷积核的requires_grad为False
            self.square_conv.weight.requires_grad = False
            self.square_conv.bias.requires_grad = False

            self.mmlist_square_conv = torch.nn.ModuleList((DiverseBranchBlockNOAct(in_channels=in_channels,
                                                                                   out_channels=out_channels,
                                                                                   kernel_size=kernel_size,
                                                                                   stride=stride,
                                                                                   padding=padding,
                                                                                   dilation=dilation,
                                                                                   groups=groups,
                                                                                   ) for _ in range(n)))
            for _ in range(n):
                self.mmlist_square_conv.append(DeepDiverseBranchBlockNOAct(in_channels=in_channels,
                                                                                   out_channels=out_channels,
                                                                                   kernel_size=kernel_size,
                                                                                   stride=stride,
                                                                                   padding=padding,
                                                                                   dilation=dilation,
                                                                                   groups=groups,
                                                                                   ))
            # self.mmlist_square_bn=torch.nn.ModuleList(nn.BatchNorm2d(num_features=out_channels, affine=use_affine)  for _ in range(n))
            # self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            # Used to check whether the specified padding value is large enough
            # to ensure that the convolution operation can be applied to
            # all pixels in the input tensor without exceeding its boundaries
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            # Vertical convolution(3x1) during training
            self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Horizontal convolution(1x3) during training
            self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Batch normalization for vertical convolution
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)
            # Batch normalization for horizontal convolution
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)

    def forward(self, input):
        if self.deploy:
            return self.nonlinear(self.fused_conv(input))
        else:
            square_outputs_list = []
            # for square_conv,square_bn in zip(self.mmlist_square_conv,self.mmlist_square_bn):
            #     square_outputs_list.append(square_bn(square_conv(input)))
            for square_conv in self.mmlist_square_conv:
                square_outputs_list.append(square_conv(input))

            # 使用torch.stack将张量堆叠在一起
            stacked_tensor = torch.stack(square_outputs_list)

            # 使用torch.sum计算张量的和
            square_outputs = torch.sum(stacked_tensor, dim=0)

            # square_outputs=self.square_bn(square_outputs)
            # square_outputs=self.nonlinear(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + (horizontal_outputs)
            return self.nonlinear(result)
            # return result

    def switch_to_deploy(self):

        # Set deploy flag to True
        self.deploy = True
        # Get the fused kernel and bias

        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        # Define the fused convolutional layer

        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels,
                                    out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size,
                                    stride=self.square_conv.stride,
                                    padding=self.square_conv.padding,
                                    dilation=self.square_conv.dilation,
                                    groups=self.square_conv.groups,
                                    bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        # Delete the original convolutional and batch normalization layers
        self.__delattr__('square_conv')
        # self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.__delattr__('mmlist_square_conv')
        # self.__delattr__('square_conv_dbb_sum')
        # Set the weight and bias of the fused convolutional layer
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def get_equivalent_kernel_bias(self):
        '''
        Used to calculate the equivalent kernel and bias of
        the fused convolution layer in deploy mode
        '''
        # Fuse batch normalization with convolutional weights and biases
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        self._fuse_dbb_conv()
        # square_k, square_b =self._fuse_bn_tensor(self.square_conv, self.square_bn)
        square_k, square_b = self.square_conv.weight, self.square_conv.bias

        # Add the fused horizontal and vertical kernels to the center of the square kernel
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        # Return the square kernel and the sum of the biases for the three convolutional layers
        return square_k, hor_b + ver_b + square_b

    def _fuse_dbb_conv(self):
        self.square_conv.weight.data = torch.zeros_like(self.square_conv.weight.data)
        self.square_conv.bias.data = torch.zeros_like(self.square_conv.bias.data)
        for dbb_conv in self.mmlist_square_conv:
            dbb_conv.switch_to_deploy()
            self.square_conv.weight.data = torch.add(self.square_conv.weight.data, dbb_conv.dbb_reparam.weight.data)
            self.square_conv.bias.data = torch.add(self.square_conv.bias.data, dbb_conv.dbb_reparam.bias.data)
            # self.square_conv.weight.data = torch.add(self.square_conv.weight.data, dbb_conv.weight.data)
            # self.square_conv.bias.data = torch.add(self.square_conv.bias.data, dbb_conv.bias.data)
            # self.square_conv.weight.data  +=dbb_conv.dbb_reparam.weight.data
            # self.square_conv.bias.data  += dbb_conv.dbb_reparam.bias.data

    def _fuse_bn_tensor(self, conv, bn):
        '''
        Fusing Conv and BN
        '''
        # Calculate standard deviation
        std = (bn.running_var + bn.eps).sqrt()
        # Calculate the tensor for fusing batch normalization into convolution
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        # Return the fused weight and bias tensors
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''
        Used to add an asymmetric kernel to the center of a square kernel
        square_kernel : the square kernel to which the asymmetric kernel will be added
        asym_kernel   : the asymmetric kernel that will be added to the square kernel
        '''
        # Get the height and width of the asymmetric kernel
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        # Get the height and width of the square kernel
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        # Add the asymmetric kernel to the center of the square kernel
        square_kernel[:,
        :,
        square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel




class ACBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros',
                 deploy=False, use_affine=True):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        padding = autopad(kernel_size, None, dilation)
        # print(padding)
        if deploy:
            # # Fused convolution(3x3) during inference (re-parameterization)
            self.fused_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=True,
                                        padding_mode=padding_mode)
        else:
            # Square convolution(3x3) during training
            self.square_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            # Used to check whether the specified padding value is large enough
            # to ensure that the convolution operation can be applied to
            # all pixels in the input tensor without exceeding its boundaries
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            # Vertical convolution(3x1) during training
            self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Horizontal convolution(1x3) during training
            self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            # Batch normalization for vertical convolution
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)
            # Batch normalization for horizontal convolution
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
                                         affine=use_affine)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result

    def switch_to_deploy(self):

        # Set deploy flag to True
        self.deploy = True
        # Get the fused kernel and bias
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        # Define the fused convolutional layer
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels,
                                    out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size,
                                    stride=self.square_conv.stride,
                                    padding=self.square_conv.padding,
                                    dilation=self.square_conv.dilation,
                                    groups=self.square_conv.groups,
                                    bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        # Delete the original convolutional and batch normalization layers
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        # Set the weight and bias of the fused convolutional layer
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def get_equivalent_kernel_bias(self):
        '''
        Used to calculate the equivalent kernel and bias of
        the fused convolution layer in deploy mode
        '''
        # Fuse batch normalization with convolutional weights and biases
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        # Add the fused horizontal and vertical kernels to the center of the square kernel
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        # Return the square kernel and the sum of the biases for the three convolutional layers
        return square_k, hor_b + ver_b + square_b

    def _fuse_bn_tensor(self, conv, bn):
        '''
        Fusing Conv and BN
        '''
        # Calculate standard deviation
        std = (bn.running_var + bn.eps).sqrt()
        # Calculate the tensor for fusing batch normalization into convolution
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        # Return the fused weight and bias tensors
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''
        Used to add an asymmetric kernel to the center of a square kernel
        square_kernel : the square kernel to which the asymmetric kernel will be added
        asym_kernel   : the asymmetric kernel that will be added to the square kernel
        '''
        # Get the height and width of the asymmetric kernel
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        # Get the height and width of the square kernel
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        # Add the asymmetric kernel to the center of the square kernel
        square_kernel[:,
        :,
        square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
