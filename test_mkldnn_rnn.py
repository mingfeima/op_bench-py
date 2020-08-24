import torch
from torch import nn
import copy

def cmp(t1, t2, debug=False):
    if debug:
        print(t1.size(), 'sum: {:.6f}'.format(t1.sum().item()))
        print(t2.size(), 'sum: {:.6f}'.format(t2.sum().item()))
    return torch.allclose(t1, t2, atol=1e-7)

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


for bidirectional in [True, False]:
    for bias in [True, False]:
        for batch_first in [True, False]:
            test_lstm_inference(35, 8, 10, 20, 3, bidirectional, bias, batch_first)
