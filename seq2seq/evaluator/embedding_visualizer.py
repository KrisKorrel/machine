from __future__ import print_function, division

import torch
import torchtext
import torch.autograd as autograd

import matplotlib.pyplot as plt

import seq2seq


class EmbeddingVisualizer(object):
    """ Class to visualize the embeddings of the encder
    """

    def __init__(self, save_dir):
        self.fig_index = 0
        self.save_dir = save_dir

    def plot(self, model, data):
        all_source_tokens = data.fields[seq2seq.src_field_name].vocab.freqs.keys()
        string_to_idx = data.fields[seq2seq.src_field_name].vocab.stoi

        for source_token in all_source_tokens:
            idx = string_to_idx[source_token]
            idx_tensor = autograd.Variable(torch.LongTensor([idx]))
            embedding = model.encoder.return_embeddings(idx_tensor)

            if source_token in ['jump', 'run', 'walk', 'look']:
                color = 'green'
            elif source_token in ['left', 'right']:
                color = 'blue'
            elif source_token in ['after']:
                color = 'red'
            elif source_token in ['and']:
                color = 'orange'
            elif source_token in ['twice', 'thrice']:
                color = 'black'
            elif source_token in ['around']:
                color = 'grey'
            elif source_token in ['turn']:
                color = 'purple'
            elif source_token in ['opposite']:
                color = 'brown'
            else:
                print(source_token)

            x, y = embed.data.cpu().numpy()[0]
            print("MOET 2 ZIJN")

            plt.scatter(x, y, marker='x', color=color, s=20)
            plt.text(x, y, source_token, color=color)

        plt.savefig('{}_{}'.format(self.save_dir, self.fig_index))
        plt.clf()

        self.fig_index += 1