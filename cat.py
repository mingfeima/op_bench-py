import torch
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
total_times = 10 # seconds

# cat(inputs, dim)
#   number of inputs: N
#   input size: [H, W]
def run_single_test(H, W, N, dim, contiguous=True):
    inputs = []
    for k in range(N):
        if contiguous:
            inputs.append(torch.randn(H, W))
        else:
            inputs.append(torch.randn(H, W + 16).narrow(1, 0, W))

    for i in range(warmups):
        output = torch.cat(inputs, dim)

    ttime = 0
    iters = 0
    while (ttime < total_times):
        t1 = time()
        output = torch.cat(inputs, dim)
        t2 = time()
        ttime = ttime + t2 - t1
        iters = iters + 1
    
    throughput = H * W * N * 4 * iters / ttime * 1e-9 

    print("input size: [{}, {}]; input number: [{}]; {}; thoughput: GB/s = {:.3f}".format(
          H, W, N, 'contiguous' if contiguous else 'non-contiguous', throughput))


def benchmark():
    for contiguous in [True, False]:
        for ninputs in [2, 16]:
            run_single_test(64, 1000, ninputs, 0, contiguous)
            run_single_test(128, 1000, ninputs, 0, contiguous)
            run_single_test(64, 10000, ninputs, 0, contiguous)
            run_single_test(128, 10000, ninputs, 0, contiguous)

benchmark()

def validate(use_bfloat=False):
    t1 = torch.randn(2, 5)
    t2 = torch.randn(2, 5)
    t3 = torch.randn(3, 5)

    if use_bfloat:
        t1, t2, t3 = t1.bfloat16(), t2.bfloat16(), t3.bfloat16()

    o1 = torch.cat([t1, t2], 0)
    o2 = torch.cat([t1, t3], 0)

    print('tensor 1: ', t1)
    print('tensor 2: ', t2)
    print('tensor 3: ', t3)
    print('cat 1&2: ', o1)
    print('cat 1&3: ', o2)

#validate()
#validate(True)
