import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

from qtorch.quant import fixed_point_quantize, quantizer
from qtorch import FixedPoint

def shift(x, ceil=False):
    max_entry = x.abs().max()
    if ceil:
        return x/2.**torch.ceil(torch.log2(max_entry))
    else:
        return x/2.**torch.round(torch.log2(max_entry))

def QW(x, bits, scale=1.0):
    y = fixed_point_quantize(x, FixedPoint(wl=bits, fl=bits-1, clamp=True, symmetric=True), rounding="nearest")
    # per layer scaling
    if scale>1.8: y /= scale
    return y

def QG(x, bits_G, bits_R, lr):
    # max_entry = x.abs().max()
    # assert max_entry != 0, "QG blow"
    x = shift(x)
    grad_number = FixedPoint(wl=bits_G, fl=bits_G-1, clamp=False, symmetric=True)
    norm = fixed_point_quantize(lr*x, grad_number, rounding="stochastic")
    return norm/(2.**((bits_G-1)))

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E, name="", writer=None):
        super(WAGEQuantizer, self).__init__()
        self.name = name
        self.writer = writer
        self.activate_number = FixedPoint(wl=bits_A, fl=bits_A-1, clamp=True, symmetric=True) if bits_A != -1 else None
        self.error_number = FixedPoint(wl=bits_E, fl=bits_E-1, clamp=True, symmetric=True) if bits_E != -1 else None
        self.quantizer = quantizer(forward_number=self.activate_number, forward_rounding="nearest",
                                   backward_number=self.error_number, backward_rounding="nearest",
                                   clamping_grad_zero=True, backward_hooks=[shift])

    def forward(self, x):
        return self.quantizer(x)

def S(bits):
    return 2.**(bits-1)

# class WAGERounding(Function):
#     @staticmethod
#     def forward(self, x, bits_A, bits_E, optional, writer=None):
#         self.optional = optional
#         self.bits_E = bits_E
#         self.writer = writer

#         if bits_A == -1:
#             ret = x
#             self.mask = torch.zeros_like(x).byte()
#         else:
#             x = fixed_point_quantize(x, FixedPoint(wl=bits_A, fl=bits_A-1, clamp=False, symmetric=True), "nearest")
#             t_max = 1- 1./S(bits_A)
#             t_min = -1 + 1./S(bits_A)
#             mask = (x > t_max) + (x < t_min)
#             ret = torch.clamp(x, t_min, t_max)
#             self.mask = mask

#         return ret

#     @staticmethod
#     def backward(self, grad_output):
#         if self.bits_E == -1: return grad_output, None, None, None

#         if self.needs_input_grad[0]:
#             try:
#                 error_number = FixedPoint(wl=self.bits_E, fl=self.bits_E-1, clamp=True, symmetric=True)
#                 grad_input = fixed_point_quantize(shift(grad_output), error_number,"nearest").masked_fill_(self.mask, 0)
#                 # grad_input = QE(grad_output, self.bits_E).masked_fill_(self.mask, 0)
#                 # grad_input = QE(grad_output, self.bits_E)
#             except AssertionError as e:
#                 print("="*80)
#                 print("Error backward:%s"%self.optional)
#                 print("-"*80)
#                 print(grad_output.max())
#                 print(grad_output.min())
#                 print("="*80)
#                 raise e
#         else:
#             grad_input = grad_output

#         return grad_input, None, None, None, None

# quantize_wage = WAGERounding.apply
# class WAGEQuantizer(Module):
#     def __init__(self, bits_A, bits_E, name="", writer=None):
#         super(WAGEQuantizer, self).__init__()
#         self.bits_A = bits_A
#         self.bits_E = bits_E
#         self.name = name
#         self.writer = writer

#     def forward(self, x):
#         y = quantize_wage(x, self.bits_A, self.bits_E, self.name)
#         if self.writer is not None:
#             self.writer.add_histogram(
#                     "activation-before/%s"%self.name, x.clone().cpu().data.numpy())
#             self.writer.add_histogram(
#                     "activation-after/%s"%self.name, y.clone().cpu().data.numpy())
#         return y
