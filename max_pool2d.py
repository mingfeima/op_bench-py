import torch
from time import time
from copy import deepcopy


def benchmark(input_size, iters):
    warmups = int(iters/100)

    input = torch.randn(input_size)
    model = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    for i in range(warmups):
        output = model(input)

    t1 = time()
    for i in range(iters):
        output = model(input)
    t2 = time()
    ttime = (t2 - t1) / iters * 1000

    print('MaxPool2d(contiguous): input_size', input_size, 'time: {:.3f} ms'.format(ttime))

    input2 = input.to(memory_format=torch.channels_last)
    model2 = deepcopy(model).to(memory_format=torch.channels_last)

    for i in range(warmups):
        output2 = model2(input2)

    t3 = time()
    for i in range(iters):
        output2 = model2(input2)
    t4 = time()
    ttime = (t4 - t3) / iters * 1000
    print('MaxPool2d(channels_last): input_size', input_size, 'time: {:.3f} ms'.format(ttime))


benchmark([1, 64, 112, 112], 2000)
benchmark([128, 64, 112, 112], 500)

### smoke test

def is_same(t1, t2, msg):
    t2_contig = t2.contiguous()
    max_diff = (t1 - t2_contig).abs().max()
    print(msg, ': size: ', t2.size(), '; stride(): ', t2.stride(),
          '; cl: ', t2.is_contiguous(memory_format=torch.channels_last),
          'sum1: ', t1.sum().item(), '; sum2: ', t2.sum().item(), '; max diff: ', max_diff.item())


def test_channels_last(input_size):
    input = torch.randn(input_size)
    input2 = input.to(memory_format=torch.channels_last)

    input.requires_grad_()
    input2.requires_grad_()

    model = torch.nn.MaxPool2d((3, 2), stride=(2, 1), return_indices=True)
    model2 = deepcopy(model).to(memory_format=torch.channels_last)

    output, indice = model(input)
    output2, indice2 = model2(input2)

    grad_output = torch.randn(output.size())
    grad_output2 = grad_output.to(memory_format=torch.channels_last)

    output.backward(grad_output)
    output2.backward(grad_output2)

    grad_input = input.grad
    grad_input2 = input2.grad

    is_same(output, output2, 'output')
    is_same(indice, indice2, 'indice')
    is_same(grad_input, grad_input2, 'grad_input')


#test_channels_last([10, 15, 5, 5])
