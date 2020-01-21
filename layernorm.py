import argparse
import torch
from torch import nn
from time import time

N = 128
T = 128
C = 1024
iters = 1000

parser = argparse.ArgumentParser(description='BatchNorm2d')
parser.add_argument('--train', action='store_true', default=False,
                    help='do training')
args = parser.parse_args()

def run_single_test(n, t, c, elementwise_affine=True):
    input = torch.randn(n, t, c)
    grad_output = torch.randn(n, t, c)
    m = nn.LayerNorm(c, elementwise_affine=elementwise_affine)
    if args.train:
        m.train()
    else:
        m.eval()

    for i in range(int(iters/10)):
        output = m(input)

    tfwd, tbwd = 0, 0

    for i in range(iters):
        t1 = time()
        if args.train:
            input.requires_grad_()
            output = m(input)
        else:
            with torch.no_grad():
                output = m(input)
        t2 = time()
        if args.train:
            output.backward(grad_output)
        t3 = time()
        tfwd += t2 - t1
        tbwd += t3 - t2

    print('LayerNorm size: [{},{},{}], elementwise_affine={}'.
          format(n,t,c, ("True" if elementwise_affine else "False")))
    if args.train:
        print("training forward: {:.2f} ms".format(tfwd / iters * 1000))
        print("training backward: {:.2f} ms".format(tbwd / iters * 1000))
    else:
        print("inference: {:.2f} ms".format(tfwd / iters * 1000))

#run_single_test(N, T, C)

def validate():
    x1 = torch.randn(8, 128, 768).requires_grad_()
    dy1 = torch.randn(8, 128, 768)
    m1 = nn.LayerNorm(768)
    m1.train()

    x2, dy2 = x1.cuda(), dy1.cuda()
    m2 = m1.cuda()

    y1 = m1(x1)
    y1.backward(dy1)
    y2 = m2(x2)
    y2.backward(dy2)

    grad_beta = grad_output.view(-1, C).sum(dim=0)
    diff_beta = grad_beta - m.bias.grad.data

    diff_y = y1 - y2.cpu()
    diff_dgamma = m1.weight.grad - m2.weight.grad.cpu()
    diff_dbeta = m1.bias.grad - m2.bias.grad.cpu()

    print('output: ', diff_y.abs().max())
    print('dgamma: ', diff_dgamma.abs().max())
    print('dbeta: ', diff_dbeta.abs().max())

validate()

