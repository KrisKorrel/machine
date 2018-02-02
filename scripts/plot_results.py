import os
import argparse
import logging

import torch
import torchtext

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

import matplotlib.pyplot as plt

from collections import OrderedDict

for max_train_length in [16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]:
    top_checkpoints_dir = os.path.join('checkpoints_exp_increasing_lengths', 'train_max_{}'.format(max_train_length))
    checkpoint_dirs = [os.path.join(top_checkpoints_dir, subdir) for subdir in os.listdir(
        top_checkpoints_dir) if os.path.isdir(os.path.join(top_checkpoints_dir, subdir))]
    checkpoint_dirs = [os.path.join(top, sub) for top in checkpoint_dirs for sub in os.listdir(top)]


    test_lengths = [tl for tl in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]]
    checkpoint_i = 0
    sum_test_acc_seq = {}
    sum_test_acc_word = {}

    for checkpoint_dir in checkpoint_dirs:
        run = checkpoint_dir[52:55]
        
        logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_dir)))
        checkpoint = Checkpoint.load(checkpoint_dir)
        seq2seq = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab

        plt.figure(1, figsize=(12,9))
        plt.figure(2, figsize=(12,9))

        for test_length in test_lengths:
            ############################################################################
            # Prepare dataset and loss
            src = SourceField()
            tgt = TargetField()
            src.vocab = input_vocab
            tgt.vocab = output_vocab
            max_len = 50

            def len_filter(example):
                return len(example.src) <= max_len and len(example.tgt) <= max_len

            # generate test set
            test = torchtext.data.TabularDataset(
                path=os.path.join('data', 'CLEANED-SCAN', 'length_split', 'single_lengths', str(test_length), 'tasks_test.txt'), format='tsv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=len_filter
            )

            # Prepare loss
            weight = torch.ones(len(output_vocab))
            pad = output_vocab.stoi[tgt.pad_token]
            loss = Perplexity(weight, pad)
            if torch.cuda.is_available():
                loss.cuda()

            #################################################################################
            # Evaluate model on test set

            evaluator = Evaluator(loss=loss, batch_size=128)
            loss, accuracy, seq_accuracy = evaluator.evaluate(seq2seq, test)

            print("Loss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))

            if test_length > max_train_length:
                color='green'
                label='test'
            else:
                color='orange'
                label='train + dev'

            plt.figure(1)
            plt.bar(test_length, seq_accuracy, color=color, label=label)
            plt.figure(2)
            plt.bar(test_length, accuracy, color=color, label=label)

            if test_length not in sum_test_acc_seq:
                sum_test_acc_seq[test_length] = seq_accuracy
                sum_test_acc_word[test_length] = accuracy
            else:
                sum_test_acc_seq[test_length] += seq_accuracy
                sum_test_acc_word[test_length] += accuracy

        plt.figure(1)
        plt.ylim(0, 1)
        plt.xticks(test_lengths)
        plt.xlabel("Output sequence length")
        plt.ylabel("Sequence accuracy")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig("sequence-accuracy_max-train-{}_run-{}".format(max_train_length, run))
        plt.clf()

        plt.figure(2)
        plt.ylim(0, 1)
        plt.xticks(test_lengths)
        plt.xlabel("Output sequence length")
        plt.ylabel("Word accuracy")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig("word-accuracy_max-train-{}_run-{}".format(max_train_length, run))
        plt.clf()
        checkpoint_i += 1

    plt.figure(3, figsize=(12,9))
    for key, val in sum_test_acc_seq.items():
        if key > max_train_length:
            color='green'
            label='test'
        else:
            color='orange'
            label='train + dev'

        plt.bar(key, float(val) / checkpoint_i, color=color, label=label)
    plt.ylim(0, 1)
    plt.xticks(test_lengths)
    plt.xlabel("Output sequence length")
    plt.ylabel("Sequence accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig("average-sequence-accuracy_max-train-{}".format(max_train_length))
    plt.clf()

    plt.figure(4, figsize=(12,9))
    for key, val in sum_test_acc_word.items():
        if key > max_train_length:
            color='green'
            label='test'
        else:
            color='orange'
            label='train + dev'

        plt.bar(key, float(val) / checkpoint_i, color=color, label=label)
    plt.ylim(0, 1)
    plt.xticks(test_lengths)
    plt.xlabel("Output sequence length")
    plt.ylabel("Word accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig("average-word-accuracy_max-train-{}".format(max_train_length))
    plt.clf()