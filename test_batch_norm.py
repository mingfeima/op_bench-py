import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
import copy


def cmp(t1, t2, msg, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    res = torch.allclose(t1, t2, atol=1e-6)
    print(msg, res, "; size: ", t2.size(), "; stride: ", t2.stride(),
          "; is_channels_last: ", t2.is_contiguous(memory_format=torch.channels_last))


def test_batchnorm2d_inference_cl(n, c, h, w, contig=True):
    print("\n### test_batchnorm2d_inference_cl")
    # 1: nchw
    # 2: blocked
    # 3: nhwc
    bn1 = nn.BatchNorm2d(c)
    bn1.eval()
    # hack weight and bias to be random value;
    # otherwise all channels in weight/bias would share the same value of 1/0 by default
    bn1.weight.requires_grad_(False)
    bn1.bias.requires_grad_(False)
    bn1.weight.mul_(torch.randn(c))
    bn1.bias.add_(torch.randn(c))
    bn2 = mkldnn_utils.to_mkldnn(bn1)
    bn3 = copy.deepcopy(bn1).to(memory_format=torch.channels_last)

    input1 = torch.randn(n, c, h, w)
    if not contig:
        input1 = torch.randn(n, c, h, w + 16).narrow(3, 0, w)
    input2 = input1.to_mkldnn()
    input3 = input1.to(memory_format=torch.channels_last)

    #print("NCHW")
    output1 = bn1(input1)
    #print("blocked")
    output2 = bn2(input2).to_dense()
    #print("NHWC")
    output3 = bn3(input3)

    cmp(output1, output2, "output: ")
    cmp(output1, output3, "output: ")


def test_batchnorm2d_training_cl(n, c, h, w, contig=True):
    print("\n### test_batchnorm2d_training_cl")
    # 1: nchw
    # 2: nhwc
    bn1 = nn.BatchNorm2d(c)
    bn1.train()
    bn1.weight.requires_grad_(True)
    bn1.bias.requires_grad_(True)
    bn2 = copy.deepcopy(bn1).to(memory_format=torch.channels_last)

    input1 = torch.randn(n, c, h, w)
    if not contig:
        input1 = torch.randn(n, c, h, w + 16).narrow(3, 0, w)
    input2 = input1.to(memory_format=torch.channels_last)
    input1.requires_grad_(True)
    input2.requires_grad_(True)

    grad_output1 = torch.randn(n, c, h, w)
    if not contig:
        grad_output1 = torch.randn(n, c, h, w + 16).narrow(3, 0, w)
    grad_output2 = grad_output1.to(memory_format=torch.channels_last)

    #print("NCHW")
    output1 = bn1(input1)
    output1.backward(grad_output1)
    #print("NHWC")
    output2 = bn2(input2)
    output2.backward(grad_output2)

    grad_input1 = input1.grad
    grad_input2 = input2.grad
    grad_weight1 = bn1.weight.grad
    grad_weight2 = bn2.weight.grad
    grad_bias1 = bn1.bias.grad
    grad_bias2 = bn2.bias.grad

    #print(grad_weight1, grad_weight2)
    #print(grad_bias1, grad_bias2)
    # verify if we have the 'same' grad buffer
    #print("\ngrad_input1.data_ptr():",  hex(grad_input1.data_ptr()))
    #print("grad_weight1.data_ptr():",  hex(grad_weight1.data_ptr()))
    #print("grad_bias1.data_ptr():",  hex(grad_bias1.data_ptr()))
    #print("grad_input2.data_ptr():",  hex(grad_input2.data_ptr()))
    #print("grad_weight2.data_ptr():",  hex(grad_weight2.data_ptr()))
    #print("grad_bias2.data_ptr():",  hex(grad_bias2.data_ptr()))
    cmp(output1, output2, "output: ")
    cmp(grad_input1, grad_input2, "grad_input:")
    cmp(grad_weight1, grad_weight2, "grad_weight:")
    cmp(grad_bias1, grad_bias2, "grad_bias:")


### smoke tests:
#test_batchnorm2d_inference_cl(100, 3, 32, 32)
#test_batchnorm2d_inference_cl(100, 3, 32, 32, False)
#test_batchnorm2d_inference_cl(128, 3, 1, 1)
#test_batchnorm2d_inference_cl(100, 1, 32, 32)

#test_batchnorm2d_training_cl(100, 3, 32, 32)
#test_batchnorm2d_training_cl(100, 3, 32, 32, False)

test_batchnorm2d_training_cl(64, 100, 1, 1)
test_batchnorm2d_training_cl(64, 1000, 1, 1)
