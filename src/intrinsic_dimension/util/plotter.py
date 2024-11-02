import os

import matplotlib.pyplot as plt
from torchviz import make_dot


def plot_results(baseline, dints, performance, basedir, name, xlabel=None, ylabel=None, title=None, show_dint90=False):
    bubble_size = 100
    fig, ax = plt.subplots()

    ax.axhline(y=baseline, linestyle='-' + '-' * bool(not show_dint90), color='black', linewidth=1, label='baseline')
    if show_dint90:
        ax.axhline(y=0.9 * baseline, linestyle='--', color='black', linewidth=1, label='90% baseline')
        plt.legend(loc="best")

    ax.scatter(dints, performance, s=bubble_size, alpha=0.5, edgecolors='black', color='navy')
    ax.plot(dints, performance, linestyle='-', color='navy', linewidth=1)

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    # save plot
    if not os.path.exists(os.path.join(basedir, "plot")):
        os.mkdir(os.path.join(basedir, "plot"))

    plt.savefig(os.path.join(basedir, "plot/{}.png".format(name)))


def plot_model(model, name, basedir, batch, loss_fn=None, labels=None):
    # save plot
    if not os.path.exists(os.path.join(basedir, "plot")):
        os.mkdir(os.path.join(basedir, "plot"))

    logits = model(batch)
    if loss_fn is not None:
        if labels is None:
            raise AttributeError("reference labels are required if loss_fn is provided for computing the loss")
        loss = loss_fn(logits, labels)
        make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(outfile=f"{basedir}/plot/{name}.png")
    else:
        make_dot(logits, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(outfile=f"{basedir}/plot/{name}.png")
