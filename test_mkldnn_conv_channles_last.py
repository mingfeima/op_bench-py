import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
import copy


def cmp(t1, t2, msg, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    res = torch.allclose(t1, t2, atol=5e-7)
    print(msg, res, "; size: ", t2.size(), "; stride: ", t2.stride(),
          "; is_channels_last: ", t2.is_contiguous(memory_format=torch.channels_last))


def test_conv2d_cl(n, ic, h, w, oc, kernel_size, groups=1):

    print("\n### test_conv2d_cl, groups =", groups)
    # 1: nchw
    # 2: blocked
    # 3: nhwc
    conv1 = nn.Conv2d(ic, oc, kernel_size)
    conv2 = mkldnn_utils.to_mkldnn(conv1)
    conv3 = copy.deepcopy(conv1).to(memory_format=torch.channels_last)

    input1 = torch.randn(n, ic, h, w)
    input2 = input1.to_mkldnn()
    input3 = input1.to(memory_format=torch.channels_last)

    input1.requires_grad_()
    input2.requires_grad_()
    input3.requires_grad_()

    #print("NCHW")
    output1 = conv1(input1)
    grad_output1 = torch.randn(output1.size()) * 1e-4
    grad_output3 = grad_output1.clone()
    output1.backward(grad_output1)
    #print("blocked")
    output2 = conv2(input2).to_dense()
    #print("NHWC")
    output3 = conv3(input3)
    output3.backward(grad_output3)

    grad_input1 = input1.grad
    grad_input3 = input3.grad
    grad_weight1 = conv1.weight.grad
    grad_weight3 = conv3.weight.grad
    grad_bias1 = conv1.bias.grad
    grad_bias3 = conv3.bias.grad
    #print("### output3.data_ptr(): ", output3.data_ptr())
    #print("### grad_input3.data_ptr(): ", grad_input3.data_ptr())
    #print("### grad_weight1.data_ptr(): ", grad_weight1.data_ptr())
    #print("### grad_bias1.data_ptr(): ", grad_bias1.data_ptr())
    #print("### grad_weight3.data_ptr(): ", grad_weight3.data_ptr())
    #print("### grad_bias3.data_ptr(): ", grad_bias3.data_ptr())

    ### note: autograd has compatibility impl for channels last
    ### need to verify if the backward output is the original
    ### buffer of mkldnn output, aka. check .data_ptr()

    cmp(output1, output2, "output: ".format(groups))
    cmp(output1, output3, "output: ".format(groups))
    cmp(grad_input1, grad_input3, "grad_input: ")
    cmp(grad_weight1, grad_weight3, "grad_weight: ")
    cmp(grad_bias1, grad_bias3, "grad_bias: ")


def test_conv2d_cl_weight_prepacking(n, ic, h, w, oc, kernel_size, groups=1):

    print("\n### test_conv2d_cl_weight_prepacking, groups =", groups)
    # 1: nchw
    # 2: nchw (weight prepacked)
    # 3: nhwc
    # 4: nhwc (weight prepacked)
    conv1 = nn.Conv2d(ic, oc, kernel_size)
    conv2 = mkldnn_utils.to_mkldnn(conv1)
    conv3 = copy.deepcopy(conv1).to(memory_format=torch.channels_last)
    conv4 = mkldnn_utils.to_mkldnn(conv3)

    input1 = torch.randn(n, ic, h, w)
    input2 = input1.clone()
    input3 = input1.to(memory_format=torch.channels_last)
    input4 = input3.clone()

    print("### nchw")
    output1 = conv1(input1)
    print("### nchw (weight prepacked)")
    output2 = conv2(input2)
    print("### nhwc")
    output3 = conv3(input3)
    print("### nhwc (weight prepacked)")
    output4 = conv4(input4)

    cmp(output1, output2, "output: ".format(groups))
    cmp(output1, output3, "output: ".format(groups))
    cmp(output1, output4, "output: ".format(groups))


### smoke tests:
#test_conv2d_cl(128, 3, 32, 32, 10, 3)
#test_conv2d_cl(128, 8, 32, 32, 10, 3, 2)
test_conv2d_cl_weight_prepacking(128, 16, 32, 32, 64, 3)


