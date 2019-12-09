import torch
from time import time
from math import sqrt

torch.manual_seed(0)
warmups = 100 # iterations
total_times = 10 # seconds

# input.uniform_(a, b)
#   input size: [N]
def run_single_test(N):
    input = torch.randn(N)

    for i in range(warmups):
        input.uniform_(0, 1)

    ttime = 0
    iters = 0
    while (ttime < total_times):
        t1 = time()
        input.uniform_(0, 1)
        t2 = time()
        ttime = ttime + t2 - t1
        iters = iters + 1
    
    throughput = N * iters / ttime * 1e-9

    print("input size: [{}]; thoughput: {:.3f} * 1e9 numbers processed per second".format(
          N, throughput))


def benchmark():
    run_single_test(1000)
    run_single_test(10000)
    run_single_test(100000)
    run_single_test(1000000)

benchmark()

def validate(use_bfloat=False):
    input1 = torch.randn(10000, dtype=torch.float32)
    input2 = torch.randn(10000, 2, dtype=torch.float32)[:, 0]
    input3 = torch.randn(10000, dtype=torch.float64)
    input4 = torch.randn(10000, 2, dtype=torch.float64)[:, 0]

    def compare(t, a, b):
        ### theoratical moments
        ### https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
        tM = (b + a) / 2
        tD = (b - a) * (b - a) / 12
        tQ = (b - a) * (b - a) * (b - a) * (b - a) / 80

        ### sample moments
        t2 = t * t
        sM = t.mean()
        sD = t2.mean()- sM * sM # VAR(X) = E(X*X) - E(X)*E(X)

        ### compare
        n = t.numel()
        tD2 = tD * tD
        s=((tQ-tD2)/n)-(2*(tQ-2*tD2)/(n*n))+((tQ-3*tD2)/(n*n*n))

        DeltaM=(tM-sM)/sqrt(tD/n)
        DeltaD=(tD-sD)/sqrt(s)
        print("### testing on {} {} tensors ###".format(("contiguous" if t.is_contiguous() else "non-contiguous"), t.dtype))
        if (abs(DeltaM)>3.0 or abs(DeltaD)>3.0):
            print("FAIL: sample moments (mean={:.6f}, variance={:.6f}) disagree with theory (mean={:.6f}, variance={:.6f})"
                  .format(sM, sD, tM, tD));
        else:
            print("PASS: Sample moments (mean={:.6f}, variance={:.6f}) agree with theory (mean={:.6f}, variance={:.6f})"
                  .format(sM, sD, tM, tD));

    for t in [input1, input2, input3, input4]:
        compare(t.uniform_(), 0, 1)


validate()
