import torch
from torch import nn
import copy

def cmp(t1, t2, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    return torch.allclose(t1, t2, atol=5e-7)

def test_lstm_inference(T, N, I, H, L, bidirectional=False, bias=False, batch_first=False):
    D = 2 if bidirectional else 1
    input = torch.randn(N, T, I) if batch_first else torch.randn(T, N, I)
    hx = torch.randn(L*D, N, H)
    cx = torch.randn(L*D, N, H)

    rnn = nn.LSTM(I, H, L, bidirectional=bidirectional, bias=bias, batch_first=batch_first)
    rnn.eval()
    rnn2 = copy.deepcopy(rnn)
    rnn2.eval()

    ### mkldnn rnn
    output, (hy, cy) = rnn(input, (hx, cx))
    ### native rnn
    torch._C._set_mkldnn_enabled(False)
    output2, (hy2, cy2) = rnn2(input, (hx, cx))
    torch._C._set_mkldnn_enabled(True)

    cmp_output = cmp(output, output2)
    cmp_hy = cmp(hy, hy2)
    cmp_cy = cmp(cy, cy2)
    print("### lstm inference: bidirectional = ",
          bidirectional, "; bias = ", bias, "; batch_first = ", batch_first,
          "; output: ", cmp_output, "; hy: ", cmp_hy, "; cy: ", cmp_cy)


def test_onehidden_inference(T, N, I, H, L, mode, bidirectional=False, bias=False, batch_first=False):
    D = 2 if bidirectional else 1
    input = torch.randn(N, T, I) if batch_first else torch.randn(T, N, I)
    hx = torch.randn(L*D, N, H)

    if mode == 'gru':
        rnn = nn.GRU(I, H, L, bidirectional=bidirectional, bias=bias, batch_first=batch_first)
    elif mode == 'rnn_relu':
        rnn = nn.RNN(I, H, L, bidirectional=bidirectional, bias=bias, batch_first=batch_first, nonlinearity='relu')
    else:
        rnn = nn.RNN(I, H, L, bidirectional=bidirectional, bias=bias, batch_first=batch_first, nonlinearity='tanh')
    rnn.eval()
    rnn2 = copy.deepcopy(rnn)
    rnn2.eval()

    ### mkldnn rnn
    output, hy = rnn(input, hx)
    ### native rnn
    torch._C._set_mkldnn_enabled(False)
    output2, hy2 = rnn2(input, hx)
    torch._C._set_mkldnn_enabled(True)

    cmp_output = cmp(output, output2)
    cmp_hy = cmp(hy, hy2)
    print("### {} inference: bidirectional = ".format(mode),
          bidirectional, "; bias = ", bias, "; batch_first = ", batch_first,
          "; output: ", cmp_output, "; hy: ", cmp_hy)


print("\n LSTM: ")
for bidirectional in [True, False]:
    for bias in [True, False]:
        for batch_first in [True, False]:
            test_lstm_inference(35, 8, 10, 20, 3, bidirectional, bias, batch_first)

print("\n GRU: ")
for bidirectional in [True, False]:
    for bias in [True, False]:
        for batch_first in [True, False]:
            test_onehidden_inference(35, 8, 10, 20, 3, 'gru', bidirectional, bias, batch_first)

print("\n RNN_RELU: ")
for bidirectional in [True, False]:
    for bias in [True, False]:
        for batch_first in [True, False]:
            test_onehidden_inference(35, 8, 10, 20, 3, 'rnn_relu', bidirectional, bias, batch_first)

print("\n RNN_TANH: ")
for bidirectional in [True, False]:
    for bias in [True, False]:
        for batch_first in [True, False]:
            test_onehidden_inference(35, 8, 10, 20, 3, 'rnn_tanh', bidirectional, bias, batch_first)
