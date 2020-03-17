import argparse

import torch
from torch import nn
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
iters = 1000 # iterations

S = 2 # scale_factor

tests = {
    'upsample_nearest1d' : [[64, 9, 512], 'linear', 'ncw', int(iters * 2)],
    'upsample_nearest2d' : [[8, 9, 64, 64], 'bilinear', 'nchw', iters],
    'upsample_nearest3d' : [[8, 9, 32, 32, 32], 'trilinear', 'ncdhw', int(iters / 10)],
    #'upsample_nearest2d' : [[8, 64, 64, 9], 'bilinear', 'nhwc', iters]
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

    for i in range(int(niters / 100)):
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


def test_linear1d():
    input1 = torch.arange(1, 4, dtype=torch.float32).view(1, 1, -1)
    input2 = input1.clone()
    input1.requires_grad_()
    input2.requires_grad_()
    m1 = nn.Upsample(scale_factor=2, mode='linear')
    m2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    output1 = m1(input1)
    output2 = m2(input2)
    grad_output1 = torch.arange(1, output1.numel() + 1).view(output1.size()).float()
    grad_output2 = grad_output1.clone()
    output1.backward(grad_output1)
    output2.backward(grad_output2)
    ref1 = torch.Tensor([[[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.0000]]])
    ref2 = torch.Tensor([[[1.0000, 1.4000, 1.8000, 2.2000, 2.6000, 3.0000]]])
    ref3 = torch.Tensor([[[ 3.2500,  7.0000, 10.7500]]])
    ref4 = torch.Tensor([[[2.8000, 8.4000, 9.8000]]])
    print("\n\ntest_linear1d\ninput, ", input)
    print("output1, align_corners=False", output1.data, "ref: ", ref1)
    print("output2, align_corners=True",  output2.data, "ref: ", ref2)
    print("grad_input1: align_corners=False", input1.grad, "ref: ", ref3)
    print("grad_input2: align_corners=True", input2.grad, "ref: ", ref4)

def test_bilinear2d():
    input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    input2 = input1.clone()
    input1.requires_grad_()
    input2.requires_grad_()
    m1 = nn.Upsample(scale_factor=2, mode='bilinear')
    m2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    output1 = m1(input1)
    output2 = m2(input2)
    grad_output1 = torch.arange(1, output1.numel() + 1).view(output1.size()).float()
    grad_output2 = grad_output1.clone()
    output1.backward(grad_output1)
    output2.backward(grad_output2)
    ref1 = torch.Tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
                           [1.5000, 1.7500, 2.2500, 2.5000],
                           [2.5000, 2.7500, 3.2500, 3.5000],
                           [3.0000, 3.2500, 3.7500, 4.0000]]]])
    ref2 = torch.Tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                           [1.6667, 2.0000, 2.3333, 2.6667],
                           [2.3333, 2.6667, 3.0000, 3.3333],
                           [3.0000, 3.3333, 3.6667, 4.0000]]]])
    ref3 = torch.Tensor([[[[16.5000, 23.5000],
                           [44.5000, 51.5000]]]])
    ref4 = torch.Tensor([[[[17.3333, 24.0000],
                           [44.0000, 50.6667]]]])
    print("\n\ntest_bilinear2d\ninput, ", input1)
    print("output1, align_corners=False", output1.data, "\nref: ", ref1)
    print("output2, align_corners=True",  output2.data, "\nref: ", ref2)
    print("grad_input1: align_corners=False", input1.grad, "\nref: ", ref3)
    print("grad_input2: align_corners=True", input2.grad, "\nref: ", ref4)

def test_trilinear3d():
    input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 1, 2, 2)
    input2 = input1.clone()
    input1.requires_grad_()
    input2.requires_grad_()
    m1 = nn.Upsample(scale_factor=2, mode='trilinear')
    m2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    output1 = m1(input1)
    output2 = m2(input2)
    grad_output1 = torch.arange(1, output1.numel() + 1).view(output1.size()).float()
    grad_output2 = grad_output1.clone()
    output1.backward(grad_output1)
    output2.backward(grad_output2)
    ref1 = torch.Tensor([[[[[1.0000, 1.2500, 1.7500, 2.0000],
                            [1.5000, 1.7500, 2.2500, 2.5000],
                            [2.5000, 2.7500, 3.2500, 3.5000],
                            [3.0000, 3.2500, 3.7500, 4.0000]],
                           [[1.0000, 1.2500, 1.7500, 2.0000],
                            [1.5000, 1.7500, 2.2500, 2.5000],
                            [2.5000, 2.7500, 3.2500, 3.5000],
                            [3.0000, 3.2500, 3.7500, 4.0000]]]]])
    ref2 = torch.Tensor([[[[[1.0000, 1.3333, 1.6667, 2.0000],
                            [1.6667, 2.0000, 2.3333, 2.6667],
                            [2.3333, 2.6667, 3.0000, 3.3333],
                            [3.0000, 3.3333, 3.6667, 4.0000]],
                           [[1.0000, 1.3333, 1.6667, 2.0000],
                            [1.6667, 2.0000, 2.3333, 2.6667],
                            [2.3333, 2.6667, 3.0000, 3.3333],
                            [3.0000, 3.3333, 3.6667, 4.0000]]]]])
    ref3 = torch.Tensor([[[[[ 97., 111.],
                            [153., 167.]]]]])
    ref4 = torch.Tensor([[[[[ 98.6667, 112.0000],
                            [152.0000, 165.3333]]]]])
    print("\n\ntest_trilinear3d\ninput, ", input1)
    print("output1, align_corners=False", output1.data, "\nref: ", ref1)
    print("output2, align_corners=True",  output2.data, "\nref: ", ref2)
    print("grad_input1: align_corners=False", input1.grad.view(-1), "\nref: ", ref3.view(-1))
    print("grad_input2: align_corners=True", input2.grad.view(-1), "\nref: ", ref4.view(-1))


#test_linear1d()
#test_bilinear2d()
#test_trilinear3d()
