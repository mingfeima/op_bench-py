import torch
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
total_times = 10 # seconds

# input.index_select(dim=0, index)
#   input size: [N, C]
#   index size: [K]
def run_single_test(N, C, K, contiguous=True, debug = False):
    if contiguous:
        input = torch.randn(N, C)
    else:
        input = torch.randn(N, C + 16).narrow(1, 0, C)

    index =  torch.randint(0, N, [K])

    if debug:
        output = input.index_select(0, index)
        print('input', input)
        print('output', output)
        print('index', index)
        return

    for i in range(warmups):
        input.index_select(0, index)

    ttime = 0
    iters = 0
    while (ttime < total_times):
        t1 = time()
        output = input.index_select(0, index)
        t2 = time()
        ttime = ttime + t2 - t1
        iters = iters + 1
    
    throughput = K * iters / ttime * 1e-6 

    print("input size: [{}, {}]; index size: [{}]; {}; thoughput: indexs/us = {:.3f}".format(
          N, C, K, 'contiguous' if contiguous else 'non-contiguous', throughput))


def benchmark():
    for contiguous in [True, False]:
        run_single_test(10000, 128, 10000, contiguous)
        run_single_test(100000, 128, 10000, contiguous)
        run_single_test(1000000, 128, 100000, contiguous)
        run_single_test(10000000, 128, 1000000, contiguous)

benchmark()

def validate():
    run_single_test(5, 4, 3, True, True)

#validate()
