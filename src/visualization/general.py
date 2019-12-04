import numpy as np
import matplotlib.pyplot as plt


def generate_subplots(n=1):
    """
    Generates standardly sized subplots.

    :param n: Number of subplots
    :return: ax. fig, ax. ax will be unravelled.
    """
    if n <= 3:
        n_cols = n
    else:
        n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    # Get the subplots
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))

    # Unravel
    if n > 1:
        ax = ax.ravel()

    return fig, ax