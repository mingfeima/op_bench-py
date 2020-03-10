import torch
from torch import nn
from time import time
import argparse
import copy

niters_inference = 2000
niters_training = 100

tests_inference = [
    # mode, L, D, N, T, I, H
    ['lstm', 1, 1, 1, 15, 250, 200]]

tests_training = [
    # mode, L, D, N, T, I, H
    ['lstm', 1, 1, 128, 50, 1024, 1024]
    ]

def run_single_test(niters, mode, L, D, N, T, I, H, train=False):
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
        y, _ = model(x, hx)

    if train:
        dy = torch.randn(T, N, D * H)

    t1 = time()
    if train:
        for i in range(niters):
            y, _ = model(x, hx)
            y.backward(dy)
    else:
        with torch.no_grad():
            for i in range(niters):
                y, _ = model(x, hx)
    t2 = time()

    ttime = (t2 - t1) / niters * 1000

    print("{}: layers={}, dir={}, batch_size={}, time_step={}, input_size={}, hidden_size={}: time: {:.2f} ms".format(
          mode, L, D, N, T, I, H, ttime))


def benchmark():
    parser = argparse.ArgumentParser(description='rnn...')
    parser.add_argument('--train', action='store_true', default=False,
                        help='benchmark training')
    args = parser.parse_args()

    tests = tests_training if args.train else tests_inference
    niters = niters_training if args.train else niters_inference
    for test in tests:
        mode, l, d, n, t, i, h = test[0], test[1], test[2], test[3], test[4], test[5], test[6]
        run_single_test(niters, mode, l, d, n, t, i, h, args.train)


benchmark()


# test fused_lstm_cell fwd and bwd
def test1():
    n = 4
    h = 10
    g = 4 * h
    input1 = torch.randn(n, g)
    hidden1 = torch.randn(n, g)
    cx1 = torch.randn(n, h)
    input2 = input1.clone()
    hidden2 = hidden1.clone()
    cx2 = cx1.clone()

    input1.requires_grad_(True)
    hidden1.requires_grad_(True)
    cx1.requires_grad_(True)

    hy1, cy1, ws1 = torch._fused_lstm_cell(input1, hidden1, cx1)
    (hy1 + cy1).sum().backward(retain_graph=True)

    input2.requires_grad_(True)
    hidden2.requires_grad_(True)
    cx2.requires_grad_(True)

    gates = input2 + hidden2
    chunked_gates = gates.chunk(4, 1)
    ig = chunked_gates[0].sigmoid_();
    fg = chunked_gates[1].sigmoid_();
    cg = chunked_gates[2].tanh_();
    og = chunked_gates[3].sigmoid_();
    cy2 = (fg * cx2).add_(ig * cg);
    hy2 = og * cy2.tanh();
    ws2 = torch.cat([ig, fg, cg, og], dim=1)
    (hy2 + cy2).sum().backward(retain_graph=True)

    def cmp(t1, t2, msg):
        print(msg, torch.allclose(t1, t2, rtol=1e-05, atol=1e-05),
                "; t1.sum(): {:.3f}, t2.sum(): {:.3f}".format(t1.sum().item(), t2.sum().item()))

    print("\n### fused_lstm_kernel ###")
    cmp(hy1, hy2, "hy: ")
    cmp(cy1, cy2, "cy: ")
    cmp(ws1, ws2, "workspace: ")
    cmp(input1.grad, input2.grad, "input_gates.grad: ")
    cmp(hidden1.grad, hidden2.grad, "hidden_gates.grad: ")
    cmp(cx1.grad, cx2.grad, "cx.grad: ")

# test fused_lstm_cell fwd and bwd
def test2():
    n = 4
    h = 10
    g = 3 * h
    input = torch.randn(n, g)
    hidden = torch.randn(n, g)
    hx = torch.randn(n, h)

    hy1, ws1 = torch._fused_gru_cell(input, hidden, hx)

    chunked_igates = input.chunk(3, 1)
    chunked_hgates = hidden.chunk(3, 1)
    reset_gate = chunked_hgates[0].add_(chunked_igates[0]).sigmoid_();
    input_gate = chunked_hgates[1].add_(chunked_igates[1]).sigmoid_();
    new_gate = chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
    hy2 = (hx - new_gate).mul_(input_gate).add_(new_gate)

    def cmp(t1, t2, msg):
        print(msg, torch.allclose(t1, t2, rtol=1e-05, atol=1e-05),
                "; t1.sum(): {:.3f}, t2.sum(): {:.3f}".format(t1.sum().item(), t2.sum().item()))

    print("\n### fused_gru_kernel ###")
    cmp(hy1, hy2, "hy: ")

test1()
test2()
