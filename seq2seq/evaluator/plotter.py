"""Plotter."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np


class PlotAttention(object):
    """PlotAttention."""

    def __init__(self):
        """Init."""

    def _showAttention(self, input_sentence, output_words, attentions, name, colour):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.tick_right()
        cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
        cbaxes = fig.add_axes([0.15, 0.11, 0.03, 0.77])
        cbaxes.yaxis.set_ticks_position('left')

        cb = plt.colorbar(cax, cax=cbaxes, ticks=[0, 0.5, 1])
        cb.ax.yaxis.set_ticks_position('left')

        # Set up axes and labels
        ax.set_xticks(np.arange(len(input_sentence)))
        ax.set_yticks(np.arange(len(output_words)))
        ax.set_xticklabels(input_sentence, fontweight='bold', rotation='vertical')
        ax.set_yticklabels(output_words, fontweight='bold')

        # Colour ticks
        for ytick, color in zip(ax.get_yticklabels(), colour):
            ytick.set_color(color)

        # X and Y labels
        ax.set_xlabel("Input", fontweight='bold')
        ax.set_ylabel("Output", fontweight='bold')
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('top')

        # plt.show()
        # exit()
        plt.savefig("{}.png".format(name))
        plt.close(fig)

    def evaluateAndShowAttention(self, input_sequence, output_sequence, output_words,
                                 attentions, correctness_check, name=None):
        """Add colors (for exact match) and plot attention."""
        correct = correctness_check(input_sequence, output_sequence, output_words)
        colour = list(map(lambda b: 'g' if b else 'r', correct))

        self._showAttention(input_sequence, output_words, attentions, name, colour)
