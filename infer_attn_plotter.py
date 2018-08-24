"""Plot attentions."""

# TODO(Kris): Example
# python infer_attn_plotter.py --checkpoint_path baseline_blabla_run_4/acc_0.83_seq_acc_0.31_target_acc_0.31_nll_loss_1.01_s800 --output_dir test_imgs --train ../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv --test ../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv

import argparse
import logging
import numpy as np
import os
import random
import torchtext

import seq2seq
from seq2seq.dataset import AttentionField
from seq2seq.dataset import SourceField
from seq2seq.dataset import TargetField
from seq2seq.evaluator import PlotAttention
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path',
                    help='Give the checkpoint path from which to load the model', required=True)
parser.add_argument('--test', help='Path to test data', required=True)
parser.add_argument('--output_dir', help='Path to save results', required=True)
parser.add_argument('--ignore_output_eos', action='store_true',
                    help='Ignore end of sequence token during training and evaluation')
parser.add_argument('--kgrammar', action='store_true',
                    help='Indicate that we should use the k-grammar metric')
opt = parser.parse_args()


def get_data_as_tensor(data):
    """Get tensor of data."""
    # Create N x 2 tensor. With strings of src and target for each data example
    master_data = np.zeros((len(data), 2), dtype=object)
    for i in range(len(data)):
        master_data[i, 0] = ' '.join(vars(data[i])[seq2seq.src_field_name])
        master_data[i, 1] = ' '.join(vars(data[i])[seq2seq.tgt_field_name][1:])
    return master_data


def load_model(checkpoint_path):
    """Load model."""
    logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return seq2seq, input_vocab, output_vocab


def prepare_data(data_path):
    """Prepare data."""
    src = SourceField()
    tgt = TargetField(include_eos=not opt.ignore_output_eos)
    attn = AttentionField(ignore_index=-1)

    tabular_data_fields = [('src', src), ('tgt', tgt), ('attn', attn)]
    gen_data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=tabular_data_fields
    )
    data_tensor = get_data_as_tensor(gen_data)

    return data_tensor


def plot_attention(test_data, img_path, correctness_check):
    """Plot attentions."""
    indices = list(range(len(test_data)))
    # random.shuffle(indices)

    for data_index in indices:
        print("Test example", data_index)
        input_sentence = test_data[data_index, 0]
        output_sentence = test_data[data_index, 1]
        input_sequence = input_sentence.split()
        output_sequence = output_sentence.split()

        outputs, attention = predictor.predict(input_sequence)
        output_length = len(test_data[data_index, 1])

        if False in correctness_check(input_sequence, output_sequence, outputs):
            print("Incorrect: ", data_index)

        img_filename = os.path.join(img_path, 'plot' + '{}'.format(data_index))
        attn_plotter.evaluateAndShowAttention(
            input_sequence,
            output_sequence,
            outputs,
            attention[:output_length],
            correctness_check,
            img_filename)


def correctness_check_lookup_tables(input_sequence, output_sequence, prediction_sequence):
    """Correctness check for lookup tables.

    Returns a list containing True or False for each prediction to indicate
    whether it is correct.
    """
    correct = []
    for i in range(len(output_sequence)):
        # Prediction is too short
        if i >= len(prediction_sequence):
            correct.append(False)
        else:
            correct.append(output_sequence[i] == prediction_sequence[i])

    if len(prediction_sequence) > len(output_sequence):
        for i in range(len(output_sequence), len(prediction_sequence)):
            correct.append(False)

    return correct


def correctness_check_kgrammar(input_sequence, output_sequence, prediction_sequence):
    """Correctness check for symbol rewriting.

    Returns a list containing True or False for each prediction to indicate
    whether it is correct.
    """
    grammar_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                     'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AS', 'BS', 'CS', 'DS',
                     'ES', 'FS', 'GS', 'HS', 'IS', 'JS', 'KS', 'LS', 'MS', 'NS', 'OS']

    def all_correct(grammar, prediction):
        all_correct = False

        # Check if everything falls in the same bucket, and there are no repeats
        for idx, inp in enumerate(grammar):
            vocab_idx = grammar_vocab.index(inp) + 1
            span = prediction[idx * 3:idx * 3 + 3]

            span_str = " ".join(span)
            if (not all(int(item.replace("A", "").replace("B", "").replace("C", "").split("_")[0]) == vocab_idx for item in span)
                    or (not ("A" in span_str and "B" in span_str and "C" in span_str))):
                all_correct = False
                break
            else:
                all_correct = True

        return all_correct

    grammar = input_sequence[:-1]
    prediction = prediction_sequence[:-1]

    correct = False
    if len(prediction) == 3 * len(grammar):
        correct = all_correct(grammar, prediction)

    correct = [correct for i in range(len(prediction_sequence))]

    return correct


if __name__ == '__main__':
    model, input_vocab, output_vocab = load_model(opt.checkpoint_path)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    predictor = Predictor(model, input_vocab, output_vocab)

    attn_plotter = PlotAttention()

    test_data = prepare_data(opt.test)

    opt_lengths = [len(test_data[i, 1].strip().split()) for i in range(test_data.shape[0])]
    trunc_length = max(opt_lengths)

    correctness_check = correctness_check_kgrammar \
        if opt.kgrammar \
        else correctness_check_lookup_tables

    plot_attention(test_data, opt.output_dir, correctness_check)
