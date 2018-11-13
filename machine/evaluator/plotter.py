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
        ax = ax = plt.gca()

        cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
        # cb = plt.colorbar(cax, orientation='horizontal', pad=0.1, ticks=[0, 0.5, 1], ax=ax)

        # X and Y labels
        ax.set_xlabel("Input", fontweight='bold')
        ax.set_ylabel("Output", fontweight='bold')
        ax.yaxis.set_label_position('left')
        ax.xaxis.set_label_position('bottom')
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('top')

        ax.set_xticks(np.arange(len(input_sentence)))
        ax.set_yticks(np.arange(len(output_words)))
        ax.set_xticklabels(input_sentence, fontweight='bold', rotation='vertical')
        ax.set_yticklabels(output_words, fontweight='bold')
        ax.grid(False, which='major')

        ax.set_xticks([x - 0.5 for x in ax.get_xticks()][1:], minor='true')
        ax.set_yticks([y - 0.5 for y in ax.get_yticks()][1:], minor='true')
        ax.grid(True, which='minor', linestyle='dotted')

        # Colour ticks
        for ytick, color in zip(ax.get_yticklabels(), colour):
            ytick.set_color(color)

        for axi in (ax.xaxis, ax.yaxis):
            for tic in axi.get_minor_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False

        # plt.show()
        # exit()
        plt.savefig("{}.png".format(name))
        plt.close('all')

    def evaluateAndShowAttention(self, input_sequence, output_sequence, output_words,
                                 attentions, correctness_check, name=None):
        """Add colors (for exact match) and plot attention."""
        correct = correctness_check(input_sequence, output_sequence, output_words)
        colour = list(map(lambda b: 'g' if b else 'r', correct))

        self._showAttention(input_sequence, output_words, attentions, name, colour)
