import numpy as np


def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    dim = 10
    N = 10000
    # the hidden dimension is randomly chosen from [60, 79] uniformly
    layer_dims = [np.random.randint(60, 80), 100]
    data = gen_data(dim, layer_dims, N)
    # (data, dim) is a (question, answer) pair
    print(data)