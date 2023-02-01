import matplotlib.pyplot as plt
import numpy as np


def get_data(X, y, side=None, is_test=False, x_dim_keep=14, common_cols=1, train_frac=0.8, data_ratio=0.5,
             dataset=None):
    if dataset is None:
        assert f'ERROR: Dataset {dataset} is not valid'

    x_train = X[:int(len(X) * train_frac)]
    y_train = y[:int(len(X) * train_frac)]
    x_val = X[int(len(X) * train_frac):]
    y_val = y[int(len(X) * train_frac):]

    split_size = int(len(x_train) * data_ratio)
    if not is_test:
        if side == 'a' or side == 'ca':
            x_train = x_train[:split_size]
            y_train = y_train[:split_size]
        elif side == 'b' or side == 'cb':
            x_train = x_train[split_size:]
            y_train = y_train[split_size:]

    if side == 'a':
        x_train = x_train[:, :, :x_dim_keep + common_cols]
        x_val = x_val[:, :, :x_dim_keep + common_cols]
        if len(x_train.shape) == 4:
            padding = ((0, 0), (0, 0), (0, x_dim_keep - common_cols), (0, 0))
        else:
            padding = ((0, 0), (0, 0), (0, x_dim_keep - common_cols))
        x_train = np.pad(x_train, pad_width=padding, mode='constant', constant_values=0)
        x_val = np.pad(x_val, pad_width=padding, mode='constant', constant_values=0)
    elif side == 'b':
        x_train = x_train[:, :, x_dim_keep - common_cols:]
        x_val = x_val[:, :, x_dim_keep - common_cols:]
        if len(x_train.shape) == 4:
            padding = ((0, 0), (0, 0), (x_dim_keep - common_cols, 0),  (0, 0))
        else:
            padding = ((0, 0), (0, 0), (x_dim_keep - common_cols, 0))
        x_train = np.pad(x_train, pad_width=padding, mode='constant', constant_values=0)
        x_val = np.pad(x_val, pad_width=padding, mode='constant', constant_values=0)
    elif side == 'ca' or side == 'cb':
        x_train = x_train[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols]
        x_val = x_val[:, :, x_dim_keep - common_cols:x_dim_keep + common_cols]
        if len(x_train.shape) == 4:
            padding = ((0, 0), (0, 0), (x_dim_keep - common_cols, x_dim_keep - common_cols), (0, 0))
        else:
            padding = ((0, 0), (0, 0), (x_dim_keep - common_cols, x_dim_keep - common_cols))
        x_train = np.pad(x_train, pad_width=padding, mode='constant', constant_values=0)
        x_val = np.pad(x_val, pad_width=padding, mode='constant', constant_values=0)
    else:
        x_train = None
        x_val = None
        assert f'ERROR: data_mode {side} must be a, b, ca or cb'

    if is_test:
        return x_train, y_train
    else:
        return (x_train, y_train), (x_val, y_val)


def plot_single_image(x):
    print(x.shape)
    plt.figure(figsize=(2, 2))
    plt.gray()
    ax = plt.subplot(1, 1, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(x)
    plt.show()


def plot_single_sources(x, y, n=1):
    print(x.shape, y.shape)
    plt.figure(figsize=(2 * n, 4))
    plt.gray()
    for i in range(0, n):
        ax = plt.subplot(2, n + 1, i + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(x[i])
    print(list(y[:n]))
    plt.show()


def plot_multiple_sources(x, y, n=1):
    plt.figure(figsize=(2 * n, 2 * len(x)))
    plt.gray()
    for i in range(0, len(x)):
        for j in range(0, n):
            ax = plt.subplot(len(x), n, j + 1 + i * n)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(x[i][j])
    plt.show()


def get_data_tabular(x, y, common_cols, mode, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    if train_frac + val_frac + test_frac != 1.0:
        assert f'ERROR: data ratio total must be 1.0, instead {train_frac + val_frac + test_frac}'
    elif mode != 'a' or mode != 'b':
        assert f'ERROR: mode must be a or b, found {mode}'

    train_cut_index = int(len(x) * train_frac)
    val_cut_index = train_cut_index + int(len(x) * val_frac)

    x_train = x[:train_cut_index]
    x_val = x[train_cut_index:val_cut_index]
    x_test = x[val_cut_index:]

    x_train_c = x[:train_cut_index][common_cols]
    x_val_c = x[train_cut_index:val_cut_index][common_cols]
    x_test_c = x[val_cut_index:][common_cols]

    y_train = y[:train_cut_index]
    y_val = y[train_cut_index:val_cut_index]
    y_test = y[val_cut_index:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), (x_train_c, y_train), (x_val_c, y_val), (x_test_c,
                                                                                                          y_test)
