import torch
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
total_times = 10 # seconds

# output = torch.masked_select(input, mask)
def run_single_test(N, C):
    input = torch.randn(N, C)
    mask = input.ge(0.0)

    for i in range(warmups):
        output = torch.masked_select(input, mask)

    ttime = 0
    iters = 0
    while(ttime < total_times):
        t1 = time()
        output = torch.masked_select(input, mask)
        t2 = time()
        ttime = ttime + t2 - t1
        iters = iters + 1

    tt = ttime * 1000 / iters
    print("input size: [{} {}]; output size: [{}]: time = {:.3f} ms".format(
          N, C, output.size()[0], tt))

def benchmark():
    #run_single_test(128, 1000)
    #run_single_test(128, 10000)
    run_single_test(128, 100000)

#benchmark()

def validate():
    input = torch.randn(3, 4)
    mask = input.ge(0.0)
    output = torch.masked_select(input, mask)
    print('bool mask')
    print('input', input)
    print('mask', mask)
    print('output', output)
    mask1 = mask.byte()
    print('byte mask')
    print('mask', mask1)
    output1 = torch.masked_select(input, mask1)
    print('output1', output1)

#validate()

def broadcast1():
    input = torch.randn(2, 5)
    mask = input.ge(0.0)[0]
    print('input', input, input.size())
    print('mask', mask, mask.size())
    output = torch.masked_select(input, mask)
    print('output', output)

def broadcast2():
    input = torch.randn(2, 5)
    mask = input.ge(0.0)
    input = input[0][0]
    print('input, ', input, input.size())
    print('mask', mask, mask.size())
    output = torch.masked_select(input, mask)
    print('output', output)

broadcast1()
broadcast2()
