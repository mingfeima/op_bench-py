import torch
from torch import nn
from time import time
import argparse
import copy

niters = 2000

tests = [
    # mode, L, D, N, T, I, H
    ['lstm', 1, 1, 1, 15, 250, 200]]

def run_single_test(mode, L, D, N, T, I, H, train=False):
    if mode == 'lstm':
        rnn = nn.LSTM
        hx_, cx_ = torch.randn(L*D, N, H), torch.randn(L*D, N, H)
        hx = (hx_, cx_)
    else:
        rnn = nn.GRU
        hx = torch.randn(L*D, N, H)

    x = torch.randn(T, N, I)
    bidirectional = False if D == 1 else True
    model = rnn(I, H, L, bidirectional=bidirectional)

    if train:
        model.train()
        x.requires_grad_()
    else:
        model.eval()

    for i in range(int(niters / 10)):
        y = model(x, hx)

    if train:
        dy = torch.randn(y.size())

    t1 = time()
    if train:
        for i in range(niters):
            y = model(x, hx)
            y.backward(dy)
    else:
        with torch.no_grad():
            for i in range(niters):
                y = model(x, hx)
    t2 = time()

    ttime = (t2 - t1) / niters * 1000

    print("{}: layers={}, dir={}, batch_size={}, time_step={}, input_size={}, hidden_size={}: time: {:.2f} ms".format(
          mode, L, D, N, T, I, H, ttime))


def benchmark():
    parser = argparse.ArgumentParser(description='rnn...')
    parser.add_argument('--train', action='store_true', default=False,
                        help='benchmark training')
    args = parser.parse_args()

    for test in tests:
        mode, l, d, n, t, i, h = test[0], test[1], test[2], test[3], test[4], test[5], test[6]
        run_single_test(mode, l, d, n, t, i, h, args.train)


benchmark()


def validate():
    l = 2
    d = 2
    n = 3
    t = 3
    i = 5
    h = 4

    x1 = torch.randn(t, n, i)
    x1.requires_grad_(True)

    m1 = nn.LSTM(i, h, l, bidirectional=True)
    m1.train()
    hx1, cx1 = torch.randn(l*d, n, h), torch.randn(l*d, n, h)
    hx1.requires_grad_(True)
    cx1.requires_grad_(True)

    x2, hx2, cx2 = x1.clone().cuda(), hx1.clone().cuda(), cx1.clone().cuda()
    m2 = copy.deepcopy(m1).cuda()
    x2.requires_grad_(True)

    y1, hn1 = m1(x1, (hx1, cx1))
    hy1, cy1 = hn1
    y1.mean().backward(retain_graph=True)

    y2, hn2 = m2(x2, (hx2, cx2))
    hy2, cy2 = hn2
    y2.mean().backward(retain_graph=True)

    def cmp(t1, t2, msg, debug=False):
        t2 = t2.cpu() if t2.is_cuda else t2
        print(msg, torch.allclose(t1, t2, rtol=1e-05, atol=1e-05))
        if debug:
            print(t1.view(-1)[0:20])
            print(t2.view(-1)[0:20])

    def cmp1(v1, v2, msg, debug):
        cmp(v1.grad.data, v2.grad.data, msg, debug)

    debug = True
    cmp(y1, y2, 'output: ', debug)
    cmp(hy1, hy2, 'hy: ', debug)
    cmp(cy1, cy2, 'cy: ', debug)


#validate()


