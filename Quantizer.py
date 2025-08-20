import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ********************* brevitas quantization ***********************
import brevitas.nn as qnn
#  self.conv1 = qnn.QuantConv2d(1, 16, 5, bias=False, weight_bit_width=self.num_bits)
#self.fc1 = qnn.QuantLinear(64 * 4 * 4, 10, bias=False, weight_bit_width=self.num_bits)

# ********************* ??(+-1) ***********************
# activation
class BinaryActivation(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        output[output == 0] = 1
        # ******************** A �� 1?0 *********************
        # output = torch.clamp(output, min=0)
        return output

    @staticmethod
    def backward(self, grad_output):
        (input,) = self.saved_tensors
        # *******************ste*********************
        grad_input = grad_output.clone()
        # ****************saturate_ste***************
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        """
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        grad_input = grad_output * grad
        """
        return grad_input

# ********************* A(??)??(??) ***********************
class ActivationQuantizer(nn.Module):
    def __init__(self, A=2):
        super(ActivationQuantizer, self).__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = BinaryActivation.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
        else:
            output = self.relu(input)
        return output


'''***************************** Weight quantization ***************************'''
class BinaryWeight(Function):
    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        output[output == 0] = 1
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* ??(+-1?0) ***********************
class TernaryWeight(Function):
    @staticmethod
    def forward(self, input):
        # **************** channel? - E(|W|) ****************
        E = torch.mean(torch.abs(input))
        # **************** ?? ****************
        threshold = E * 0.7
        # ************** W �� +-1?0 **************
        output = torch.sign(
            torch.add(
                torch.sign(torch.add(input, threshold)),
                torch.sign(torch.add(input, -threshold)),
            )
        )
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input

# ********************* 3 bit(+-3?+-2?+-1?0) ***********************
class ThreeBitQuant(Function):
    @staticmethod
    def forward(ctx, input):
        # Calculate the mean of absolute values (L1 norm divided by element count)
        m = input.data.norm(p=1).div(input.data.nelement())
        thres1 = 0.5 * m
        thres2 = 1.5 * m
        thres3 = 2.5 * m
        
        # Create quantization tensor
        Q = torch.zeros_like(input)
        # Assign quantization levels
        Q[input > thres3] = 3
        Q[(input <= thres3) & (input > thres2)] = 2
        Q[(input <= thres2) & (input > thres1)] = 1
        Q[(input <= thres1) & (input > -thres1)] = 0
        Q[(input <= -thres1) & (input > -thres2)] = -1
        Q[(input <= -thres2) & (input > -thres3)] = -2
        Q[input <= -thres3] = -3
        
        return Q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output


# ********************* W(????)??(?/??) ***********************
def meancenter_clamp_convparams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub_(mean)  # W???(C??)
    w.data.clamp_(-1.0, 1.0)  # W??
    return w



class WeightQuantizer(nn.Module): 
    def __init__(self, W=3):
        super(WeightQuantizer, self).__init__()
        self.W = W

    def binary(self, input):
        output = BinaryWeight.apply(input)
        return output

    def ternary(self, input):
        output = TernaryWeight.apply(input)
        return output
    def bit_3ry(self, input):
        output = ThreeBitQuant.apply(input)
        return output

    def forward(self, input):
        # **************************************** W binary *****************************************
        if self.W == 1:
            output = meancenter_clamp_convparams(input)
            E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
            alpha = E
            output = self.binary(output)
            output = output * alpha
        # **************************************** W Ternary *****************************************
        elif self.W == 2:
            output_fp = input.clone()
            # ************** W �� +-1?0 **************
            output, threshold = self.ternary(input)  # threshold(??)
            # **************** a(????) ****************
            output_abs = torch.abs(output_fp)
            mask_le = output_abs.le(threshold)
            mask_gt = output_abs.gt(threshold)
            output_abs[mask_le] = 0
            output_abs_th = output_abs.clone()
            output_abs_th_sum = torch.sum(output_abs_th)
            mask_gt_sum = torch.sum(mask_gt).float()
            alpha = output_abs_th_sum / mask_gt_sum  # a(????)
            # *************** W * a ****************
            output = output * alpha  # ????a(????),?????
        # **************************************** W 3-bit *****************************************
        elif self.W == 3:
            # Quantize to integer levels
            Q = self.bit_3ry(input)
            
            # Compute scaling factor alpha for the entire layer
            with torch.no_grad():
                # Optimal alpha minimizes quantization error (least squares solution)
                numerator = torch.sum(input * Q)
                denominator = torch.sum(Q * Q)
                # Avoid division by zero
                if denominator.abs() < 1e-6:
                    alpha = torch.tensor(0.0, device=input.device)
                else:
                    alpha = numerator / denominator
            
            # Apply scaling factor
            output = alpha * Q
        else:
            output = input
        return output


'''***************************** Activation quantization ***************************'''
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == "L":  # layer?(activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == "C":  # channel?(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == "FC":  # channel?(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == "L":
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.q_level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.q_level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class HistogramObserver(nn.Module):
    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(
            input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0
        )[0]
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)


# ********************* quantizers(???,??) *********************
# ??(??/??ste)
class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # ??
        if q_type == 0:
            max_val = torch.max(
                torch.abs(observer_min_val), torch.abs(observer_max_val)
            )
            min_val = -max_val
        # ???
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None
        

class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        # scale/zero_point/eps
        if self.observer.q_level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
            
        elif self.observer.q_level == "C":
            self.register_buffer("scale",torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),)
            self.register_buffer("zero_point",torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),)
            
        elif self.observer.q_level == "FC":
            self.register_buffer("scale",torch.ones((self.observer.out_channels, 1), dtype=torch.float32),)
            self.register_buffer("zero_point",torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),)
            
        self.register_buffer("eps", torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))

    def update_qparams(self):
        raise NotImplementedError

    # ??(ste)
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print("!Binary quantization is not supported !")
            assert self.bits != 1
        else:
            if not self.qaft:
                # qat, update quant_para
                if self.training:
                    if not self.union:
                        self.observer(input)  # update observer_min and observer_max
                    self.update_qparams()  # update scale and zero_point
            output = (torch.clamp(self.round(
                        input / self.scale.clone() - self.zero_point,
                        self.observer.min_val / self.scale - self.zero_point,
                        self.observer.max_val / self.scale - self.zero_point,
                        self.q_type,),self.quant_min_val,self.quant_max_val,)+ self.zero_point) * self.scale.clone()
        return output


class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:  # weight
            self.register_buffer(
                "quant_min_val",
                torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32),
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),
            )
        elif self.activation_weight_flag == 1:  # activation
            self.register_buffer(
                "quant_min_val",
                torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32),
            )
            self.register_buffer(
                "quant_max_val",
                torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32),
            )
        else:
            print("activation_weight_flag error")


# ????
class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = (
            float(self.quant_max_val - self.quant_min_val) / 2
        )  # quantized_range
        float_range = torch.max(
            torch.abs(self.observer.min_val), torch.abs(self.observer.max_val)
        )  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

class Update_QA(Function):
    @staticmethod
    def forward(ctx, x, bit, min_val, max_val):
        delta = max_val - min_val
        
        if delta == 0:
            return x  # Avoid division by zero
        
        stepSizeRatio = 2.**(-bit)
        stepSize = stepSizeRatio * delta
        index = torch.floor((x - min_val) / stepSize)
        index = torch.clamp(index, 0, (2.**bit)-1)
        y = index * stepSize + min_val
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        return grad_output, None, None, None

class QuantA(nn.Module):
    def __init__(self, a_bit):
        super(QuantA, self).__init__()
        self.a_bit = a_bit
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.init_b = True
        self.momentum = 0.1

    def update_buffers(self, x):
      with torch.no_grad():
          min_a = x.min().abs()
          max_a = x.max().abs()
          a = max(min_a, max_a)
          current_min = -a
          current_max = a
          if self.init_b:
              self.running_min.copy_(current_min)
              self.running_max.copy_(current_max)
              self.init_b = False
          else:
              self.running_min.mul_(1 - self.momentum).add_(current_min * self.momentum)
              self.running_max.mul_(1 - self.momentum).add_(current_max * self.momentum)

    def forward(self, x):
        if self.training:
            self.update_buffers(x.detach())
        with torch.no_grad():
            current_min = self.running_min.clone().detach()
            current_max = self.running_max.clone().detach()
            #self.scale = 2**(self.adc_quant-1)/current_max
        out_clamped = torch.clamp(x, current_min, current_max)
        out_q = Update_QA.apply(out_clamped, self.a_bit, current_min, current_max)
        return out_q





class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        W=2,
        quant_inference=False,
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)

    def forward(self, input):
        if not self.quant_inference:
            tnn_bin_weight = self.weight_quantizer(self.weight)
        else:
            tnn_bin_weight = self.weight
        output = F.conv2d(
            input,
            tnn_bin_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        W=2,
        quant_inference=False,
    ):
        super(QuantConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)

    def forward(self, input):
        if not self.quant_inference:
            tnn_bin_weight = self.weight_quantizer(self.weight)
        else:
            tnn_bin_weight = self.weight
        output = F.conv_transpose2d(
            input,
            tnn_bin_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        return output


def add_quant_op(module, layer_counter, layer_num, A=2, W=2, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1 and layer_counter[0] < layer_num:
                if child.bias is not None:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        W=W,
                        quant_inference=quant_inference,
                    )
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        W=W,
                        quant_inference=quant_inference,
                    )
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1 and layer_counter[0] < layer_num:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        output_padding=child.output_padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        W=W,
                        quant_inference=quant_inference,
                    )
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        output_padding=child.output_padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        W=W,
                        quant_inference=quant_inference,
                    )
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.ReLU):
            if layer_counter[0] > 0 and layer_counter[0] < layer_num:
                quant_relu = ActivationQuantizer(A=A)
                module._modules[name] = quant_relu
        else:
            add_quant_op(
                child,
                layer_counter,
                layer_num,
                A=A,
                W=W,
                quant_inference=quant_inference,
            )


def prepare(model, inplace=False, A=2, W=2, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    layer_num = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layer_num += 1
        elif isinstance(m, nn.ConvTranspose2d):
            layer_num += 1
    add_quant_op(
        model, layer_counter, layer_num, A=A, W=W, quant_inference=quant_inference
    )
    return model




class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, W=3,A=5, quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)

        
        self.a_bit = A
        #self.activation_quantizer = SymmetricQuantizer(bits=self.a_bit, observer=HistogramObserver(q_level="L"), activation_weight_flag=1, qaft=False)
        self.activation_quantizer = QuantA(a_bit=self.a_bit)
    def forward(self, input):
        quant_weight = self.weight_quantizer(self.weight)
        out = F.linear(input, quant_weight, self.bias)
        return self.activation_quantizer(out)
    

