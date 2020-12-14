import argparse
import torch
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
from time import time
import copy

iters = 1000
bs=1

parser = argparse.ArgumentParser(description='BatchNorm2d')
parser.add_argument('--train', action='store_true', default=False,
                    help='do training')
args = parser.parse_args()

def run_single_test(sz):
    n, c, h, w = bs, sz[1], sz[2], sz[3]
    niters = iters
    if args.train:
        niters = int(iters / 10)

    input = torch.randn(n, c, h, w)
    grad_output = torch.randn(n, c, h, w)
    m = nn.BatchNorm2d(c)
    if args.train:
        m.train()
    else:
        m.eval()

    # channels last
    input2 = input.clone().to(memory_format=torch.channels_last)
    m2 = copy.deepcopy(m).to(memory_format=torch.channels_last)
    grad_output2 = grad_output.clone().to(memory_format=torch.channels_last)

    # blocked
    #input3 = input.clone().to_mkldnn()
    #m3 = mkldnn_utils.to_mkldnn(m)

    for i in range(int(niters/10)):
        output = m(input)

    t1 = time()
    if args.train:
        for i in range(niters):
            input.requires_grad_()
            output = m(input)
            output.backward(grad_output)
    else:
        for i in range(niters):
            output = m(input)
    t2 = time()
    tt = (t2 - t1) / niters * 1000

    for i in range(int(niters/10)):
        output2 = m2(input2)

    t3 = time()
    if args.train:
        for i in range(niters):
            input.requires_grad_()
            output2 = m2(input2)
            output2.backward(grad_output2)
    else:
        for i in range(niters):
            output2 = m2(input2)
    t4 = time()
    tt2 = (t4 - t3) / niters * 1000

    #t5 = time()
    #for i in range(niters):
    #    output3 = m3(input3)
    #t6 = time()
    #tt3 = (t6 - t5) / niters * 1000

    print('BatchNorm size(contiguous): [{},{},{},{}]: {:.3f} ms'.format(n, c, h, w, tt))
    print('BatchNorm size(channels last): [{},{},{},{}]: {:.3f} ms'.format(n, c, h, w, tt2))
    #print('BatchNorm size(blocked: [{},{},{},{}]: {:.3f} ms'.format(n, c, h, w, tt3))


rn50_bn_sizes = [
[1, 64, 112, 112],
[1, 64, 56, 56],
[1, 256, 56, 56],
[1, 128, 56, 56],
[1, 128, 28, 28],
[1, 512, 28, 28],
[1, 256, 28, 28],
[1, 256, 14, 14],
[1, 1024, 14, 14],
[1, 256, 14, 14],
[1, 512, 14, 14],
[1, 512, 7, 7],
[1, 2048, 7, 7]
]

for sz in rn50_bn_sizes:
    run_single_test(sz)

