import argparse

import torch
from torch import nn
from time import time


torch.manual_seed(0)
warmups = 1#10 # iterations
iters = 5#100 # iterations

S = 2 # scale_factor

tests = {
    'upsample_bilinear' : [[32, 128, 64, 64], 'bicubic', 'nchw', iters],
    'upsample_bilinear_cl' : [[32, 64, 64, 128], 'bicubic', 'nhwc', iters]
}

# nn.Upsample(input, scale_factor, mode)
def run_single_test(name, input_size, scale, mode, memory_format, niters, train):
    input = torch.randn(input_size)
    if memory_format == 'nhwc':
        #input = input.contiguous(memory_format=torch.channels_last)
        input = input.permute(0, 3, 1, 2)
    if train:
        input.requires_grad_()
    
    model = nn.Upsample(scale_factor=scale, mode=mode)

    for i in range(warmups):
        output = model(input)

    grad_output = torch.randn(output.size())

    t1 = time()
    for i in range(niters):
        output = model(input)
        if train:
            output.backward(grad_output)
    t2 = time()
    ttime = t2 - t1

    print("{}: memory format: {}, input size: ".format(name, memory_format), input.size())
    print("input.is_contiguous(memory_format=torch.channels_last): ", input.is_contiguous(memory_format=torch.channels_last))
    print("input.is_contiguous(): ", input.is_contiguous())
    if train:
        print("forward + backward time per iteration: {:.3f} ms".format((ttime) / niters * 1000))
    else:
        print("forward time per iteration: {:.3f} ms".format(ttime / niters * 1000))


def benchmark():
    parser = argparse.ArgumentParser(description='upsample...')
    parser.add_argument('--train', action='store_true', default=False,
        help='benchmark training')
    args = parser.parse_args()

    for name, input in tests.items():
        input_size, mode, mformat, niters = input[0], input[1], input[2], input[3]
        run_single_test(name, input_size, S, mode, mformat, niters, args.train)

benchmark()


### Some smoke tests
def test_bicubic():
    input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    input2 = input1.clone()
    input1.requires_grad_()
    input2.requires_grad_()
    m1 = nn.Upsample(scale_factor=2, mode='bicubic')
    m2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
    output1 = m1(input1)
    output2 = m2(input2)
    grad_output1 = torch.arange(1, output1.numel() + 1).view(output1.size()).float()
    grad_output2 = grad_output1.clone()
    output1.backward(grad_output1)
    output2.backward(grad_output2)
    ref1 = torch.Tensor([[[[0.6836, 1.0156, 1.5625, 1.8945],
                           [1.3477, 1.6797, 2.2266, 2.5586],
                           [2.4414, 2.7734, 3.3203, 3.6523],
                           [3.1055, 3.4375, 3.9844, 4.3164]]]])
    ref2 = torch.Tensor([[[[1.0000, 1.3148, 1.6852, 2.0000],
                           [1.6296, 1.9444, 2.3148, 2.6296],
                           [2.3704, 2.6852, 3.0556, 3.3704],
                           [3.0000, 3.3148, 3.6852, 4.0000]]]])
    ref3 = torch.Tensor([[[[13.1016, 21.4609],
                           [46.5391, 54.8984]]]])
    ref4 = torch.Tensor([[[[17.1482, 23.8889],
                           [44.1111, 50.8518]]]])
    print("\n\ntest_bilinear2d\ninput, ", input1)
    print("output1, align_corners=False\n", output1.data, "\nref:\n", ref1)
    print("output2, align_corners=True\n",  output2.data, "\nref:\n", ref2)
    print("grad_input1: align_corners=False\n", input1.grad, "\nref:\n", ref3)
    print("grad_input2: align_corners=True\n", input2.grad, "\nref:\n", ref4)


def test_bicubic_nhwc():
    n, c, h, w = 30, 40, 50, 60
    x = torch.randn(n, c, h, w)
    input1 = x.permute(0, 3, 1, 2) # nhwc
    input2 = input1.contiguous() # nchw

    for align_corners in [True, False]:
        m = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=align_corners)
        output1 = m(input1)
        output2 = m(input2)
        diff = (output1 - output2).abs().max().item()
        print("test_bicubic, align_corners: ", align_corners, "; allclose: ", output1.allclose(output2, rtol=1e-05, atol=1e-06), "; max diff: ", diff)
        if output1.is_contiguous(memory_format=torch.channels_last):
            print("Pass: output is channels last memory format")
        else:
            print("Fail: output is contigous memory format")


#test_bicubic()
#test_bicubic_nhwc()
