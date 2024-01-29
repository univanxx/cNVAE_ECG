import matplotlib.pyplot as plt


def plot_ecg(input, num_channels):
    fig, ax = plt.subplots(num_channels, figsize=(15,7))
    for i in range(num_channels):
        ax[i].plot(input[i])
    return fig


def compare_ecgs(gt, pred):
    fig, ax = plt.subplots(gt.shape[0], 2, figsize=(15,7))
    for i in range(gt.shape[0]):
        ax[i][0].plot(gt[i])
        ax[i][1].plot(pred[i].detach())
    return fig