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
    run_single_test(128, 1000)
    run_single_test(128, 10000)
    run_single_test(128, 100000)

benchmark()
