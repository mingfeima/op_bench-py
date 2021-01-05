import torch
from time import time
from copy import deepcopy

torch.manual_seed(0)

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


#benchmark([1, 64, 112, 112], 2000)
#benchmark([128, 64, 112, 112], 500)

### smoke test

def cmp(t1, t2, msg, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    res = torch.allclose(t1, t2, atol=1e-6)
    print(msg, res, "; size: ", t2.size(), "; stride: ", t2.stride(),
          "; is_channels_last: ", t2.is_contiguous(memory_format=torch.channels_last))

def test_channels_last(input_size):
    print("### test_channels_last ###")
    input = torch.randn(input_size)
    input2 = input.to(memory_format=torch.channels_last)

    pool = torch.nn.MaxPool2d((3, 2), stride=(2, 1), return_indices=True)
    pool2 = deepcopy(pool).to(memory_format=torch.channels_last)
    unpool = torch.nn.MaxUnpool2d((3, 2), stride=(2, 1))
    unpool2= deepcopy(unpool).to(memory_format=torch.channels_last)

    output, indice = pool(input)
    output2, indice2 = pool2(input2)

    output.requires_grad_()
    output2.requires_grad_()

    output_unpool = unpool(output, indice)
    output_unpool2 = unpool2(output2, indice2)

    grad_output_unpool = torch.randn(output_unpool.size())
    grad_output_unpool2 = grad_output_unpool.to(memory_format=torch.channels_last)

    output_unpool.backward(grad_output_unpool)
    output_unpool2.backward(grad_output_unpool2)

    grad_input_unpool = output.grad
    grad_input_unpool2 = output2.grad

    #print('grad_input_unpool.data_ptr(): ', hex(grad_input_unpool.data_ptr()))
    #print('grad_input_unpool2.data_ptr(): ', hex(grad_input_unpool2.data_ptr()))

    cmp(output_unpool, output_unpool2, 'output_unpool,')
    cmp(grad_input_unpool, grad_input_unpool2, 'grad_input_unpool,')


def test_max_unpool3d():
    print("\n### test_max_unpool3d ###")
    input = torch.randn(3, 10, 32, 32, 32)
    pool = torch.nn.MaxPool3d(3, stride=1, return_indices=True)
    unpool = torch.nn.MaxUnpool3d(3, stride=1)

    output, indice = pool(input)
    output.requires_grad_()

    output_unpool = unpool(output, indice)
    grad_output_unpool = torch.randn(output_unpool.size())

    output_unpool.backward(grad_output_unpool)
    grad_input_unpool = output.grad

    ### output.sum():  1614206.25 1.697903037071228 2.365394353866577
    y = output.view(-1)
    print('output.sum(): ', output.sum().item(), y[123].item(), y[456].item())

    ### grad_input.sum(): 3630.34716796875 1.535415530204773 -1.2030938863754272
    dx = grad_input_unpool.view(-1)
    print('grad_input.sum():', grad_input_unpool.sum().item(), dx[321].item(), dx[654].item())


test_channels_last([10, 15, 5, 5])
test_max_unpool3d()
