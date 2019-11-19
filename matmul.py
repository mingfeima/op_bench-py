import torch
from time import time

warmups = 100
iters = 2000

def single_test(M, N, K):
    A = torch.randn(M, K)
    B = torch.randn(K, N)

    for i in range(warmups):
        torch.matmul(A, B)

    t1 = time()
    for i in range(iters):
        torch.matmul(A, B)
    t2 = time()

    print("M {}, K {}, N {}, time: {:.3f} ms".format(M, K, N, (t2 - t1) / iters * 1000))

single_test(128, 1000, 1000)
single_test(256, 1000, 1000)
single_test(512, 1000, 1000)
