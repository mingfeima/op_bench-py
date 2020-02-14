import torch
from time import time


torch.manual_seed(0)
warmups = 100 # iterations
total_times = 10 # seconds

# output = input.masked_fill_(mask, value)
def run_single_test(N, C, contiguous=True):
    if contiguous:
        input = torch.randn(N, C)
    else:
        input = torch.randn(N, C + 16).narrow(1, 0, C)

    mask = input.ge(0.0)

    for i in range(warmups):
        output = input.masked_fill_(mask, 0)

    ttime = 0
    iters = 0
    while(ttime < total_times):
        t1 = time()
        output = input.masked_fill_(mask, 0)
        t2 = time()
        ttime = ttime + t2 - t1
        iters = iters + 1

    tt = ttime * 1000 / iters
    print("input size: [{} {}]; contiguous: {}; time = {:.3f} ms".format(
          N, C, ("True" if contiguous else "False"), tt))

def benchmark():
    for contig in [True, False]:
        run_single_test(128, 1000, contig)
        run_single_test(256, 1000, contig)
        run_single_test(512, 1000, contig)
        run_single_test(1024, 1000, contig)

benchmark()

def validate():
    input = torch.randn(3, 4)
    mask = input.ge(0.0)
    print('bool mask')
    print('input', input)
    print('mask', mask)
    output = input.masked_fill_(mask, 0)
    print('output', output)
    mask1 = mask.byte()
    print('byte mask')
    print('mask', mask1)
    output1 = input.masked_fill_(mask1, 0)
    print('output1', output1)

#validate()
