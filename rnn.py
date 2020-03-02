import torch
from torch import nn
from time import time
import argparse

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
    for i in range(niters):
        y = model(x, hx)
        if train:
            y.backward(dy)
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





