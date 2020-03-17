import argparse

import torch
from torch import nn
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
iters = 10000 # iterations

S = 2 # scale_factor

tests = {
    #'upsample_nearest1d' : [[64, 9, 512], 'nearest', 'ncw', int(iters * 2)],
    #'upsample_nearest2d' : [[8, 9, 64, 64], 'nearest', 'nchw', iters],
    #'upsample_nearest3d' : [[8, 9, 32, 32, 32], 'nearest', 'ncdhw', int(iters / 10)],
    'upsample_nearest2d' : [[8, 64, 64, 9], 'nearest', 'nhwc', iters]
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

    fwd_t, bwd_t = 0, 0
    for i in range(niters):
        t1 = time()
        output = model(input)
        t2 = time()
        if train:
            output.backward(grad_output)
        t3 = time()
        fwd_t = fwd_t + (t2 - t1)
        bwd_t = bwd_t + (t3 - t2)

    print("{}: memory format: {}, input size: ".format(name, memory_format), input.size())
    print("input.is_contiguous(memory_format=torch.channels_last): ", input.is_contiguous(memory_format=torch.channels_last))
    print("input.is_contiguous(): ", input.is_contiguous())
    if train:
        print("forward + backward time per iteration: {:.3f} ms".format((fwd_t + bwd_t) / niters * 1000))
    else:
        print("forward time per iteration: {:.3f} ms".format(fwd_t / niters * 1000))


def benchmark():
    parser = argparse.ArgumentParser(description='upsample...')
    parser.add_argument('--train', action='store_true', default=False,
        help='benchmark training')
    args = parser.parse_args()

    for name, input in tests.items():
        input_size, mode, mformat, niters = input[0], input[1], input[2], input[3]
        run_single_test(name, input_size, S, mode, mformat, niters, args.train)

benchmark()


def validate():
    model = nn.Upsample(scale_factor=S, mode='nearest')

    input1 = torch.randn(1, 1, 4).requires_grad_()
    output1 = model(input1)
    grad_output1 = torch.arange(output1.numel()).view(output1.size()).float()
    output1.backward(grad_output1)
    print('input1: ', input1.data)
    print('output1: ', output1.data)
    print('grad_output1: ', grad_output1)
    print('grad_input1: ', input1.grad)

    input2 = torch.randn(1, 1, 2, 2).requires_grad_()
    output2 = model(input2)
    grad_output2 = torch.arange(output2.numel()).view(output2.size()).float()
    output2.backward(grad_output2)
    print('input2: ', input2.data)
    print('output2: ', output2.data)
    print('grad_output2: ', grad_output2)
    print('grad_input2: ', input2.grad)

    input3 = torch.randn(1, 1, 1, 2, 2).requires_grad_()
    output3 = model(input3)
    grad_output3 = torch.arange(output3.numel()).view(output3.size()).float()
    output3.backward(grad_output3)
    print('input3: ', input3.data)
    print('output3: ', output3.data)
    print('grad_output3: ', grad_output3)
    print('grad_input3: ', input3.grad)

#validate()
