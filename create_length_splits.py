"""
Script to make it easier to split the data set on basis of lengths
"""

import os
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--leave_outs', nargs='+', type=int, required=True)
opt = parser.parse_args()

def create_split(excluded, output_name):
    input_file_all = open(os.path.join('data', 'CLEANED-SCAN', 'tasks.txt'), 'r')

    if not os.path.exists(os.path.join('data', 'CLEANED-SCAN', 'length_split', output_name)):
        os.mkdir(os.path.join('data', 'CLEANED-SCAN', 'length_split', output_name))
    output_file_train = open(os.path.join('data', 'CLEANED-SCAN', 
        'length_split', output_name, 'tasks_train.txt'), 'w')

    n_included = 0
    n_excluded = 0

    actual_included = set()

    for line in input_file_all:
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        input_sequence_length = len(input_sequence.split())
        output_sequence_length = len(output_sequence.split())

        if output_sequence_length not in excluded:
            output_file_train.write(line)
            n_included += 1
            actual_included.add(output_sequence_length)
        else:
            n_excluded += 1

    total_number_of_tasks = n_included + n_excluded

    print("Included lengths in training set ({:.02f}%): {}".format(100.0 * n_included / total_number_of_tasks, sorted(actual_included)))
    print("Results are in '{}'".format(os.path.join('length_split', output_name)))
    print('')

if __name__ == '__main__':
    output_name = ''
    for i in opt.leave_outs:
        output_name += '{} '.format(i)
    output_name = output_name[:-1]

    create_split(excluded=opt.leave_outs + list(range(23,49)), output_name=output_name)
