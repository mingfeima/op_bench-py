import torch
from time import time
from copy import deepcopy


def benchmark(input_size, output_size, iters):
    warmups = int(iters/100)

    input = torch.randn(input_size)
    model = torch.nn.AdaptiveMaxPool2d(output_size)

    for i in range(warmups):
        output = model(input)

    t1 = time()
    for i in range(iters):
        output = model(input)
    t2 = time()
    ttime = (t2 - t1) / iters * 1000

    print('AdaptiveMaxPool2d(contiguous): input_size', input_size, '; output_size: ', output_size, 'time: {:.3f} ms'.format(ttime))

    input2 = input.to(memory_format=torch.channels_last)
    model2 = deepcopy(model).to(memory_format=torch.channels_last)

    for i in range(warmups):
        output2 = model2(input2)

    t3 = time()
    for i in range(iters):
        output2 = model2(input2)
    t4 = time()
    ttime = (t4 - t3) / iters * 1000
    print('AdaptiveMaxPool2d(channels_last): input_size', input_size, '; output_size: ', output_size, 'time: {:.3f} ms'.format(ttime))


#benchmark([1, 2048, 7, 7], [2, 2], 10000)
#benchmark([128, 2048, 7, 7], [2, 2], 1000)
#benchmark([1, 2048, 7, 7], [1, 1], 10000)
#benchmark([128, 2048, 7, 7], [1, 1], 1000)


### smoke test

def cmp(t1, t2, msg, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    res = torch.allclose(t1, t2, atol=5e-7)
    print(msg, res, "; size: ", t2.size(), "; stride: ", t2.stride(),
          "; is_channels_last: ", t2.is_contiguous(memory_format=torch.channels_last))


def test_channels_last(input_size, output_size):
    input = torch.randn(input_size)
    input2 = input.to(memory_format=torch.channels_last)

    input.requires_grad_()
    input2.requires_grad_()

    model = torch.nn.AdaptiveMaxPool2d(output_size, return_indices=True)
    model2 = deepcopy(model).to(memory_format=torch.channels_last)

    output, indices = model(input)
    output2, indices2 = model2(input2)

    grad_output = torch.randn(output.size())
    grad_output2 = grad_output.to(memory_format=torch.channels_last)

    output.backward(grad_output)
    output2.backward(grad_output2)

    grad_input = input.grad
    grad_input2 = input2.grad

    cmp(output, output2, "output:")
    cmp(indices, indices2, "indice:")
    cmp(grad_input, grad_input2, "grad_input:")
    print("### grad_input2.data_ptr(): ", hex(grad_input2.data_ptr()))

#test_channels_last([1, 1, 2, 2, ], [1, 1])

test_channels_last([16, 2048, 7, 7], [1, 1])
test_channels_last([128, 19, 4, 4], [2, 2])
#test_channels_last([2, 9, 4, 4], [2, 2])

