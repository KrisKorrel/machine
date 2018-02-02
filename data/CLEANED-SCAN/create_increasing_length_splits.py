"""
Script to make it easier to split the data set on basis of lengths
"""

import os
from collections import Counter
import numpy

def show_statistics():
    """
    Opens the file containing all data examples, counts the distribution of these
    examples in terms of length of both input and output, and prints these statistics
    """
    input_length_counter = Counter()
    output_length_counter = Counter()

    with open('tasks.txt', 'r') as all_tasks_file:
        total_number_of_tasks = 0

        for line in all_tasks_file:
            total_number_of_tasks += 1

            line = line.strip()
            input_sequence, output_sequence = line.split('\t')

            input_sequence_length = len(input_sequence.split())
            output_sequence_length = len(output_sequence.split())

            input_length_counter[input_sequence_length] += 1
            output_length_counter[output_sequence_length] += 1

        print("How often input lengths occur:")
        accumulative_input_count = 0
        for input_length in sorted(input_length_counter.keys()):
            count = input_length_counter[input_length]
            accumulative_input_count += count
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count,
                                                                                      100.0 * count / total_number_of_tasks, 100.0 * accumulative_input_count / total_number_of_tasks))
        print("")

        print("How often input lengths occur (reversed):")
        accumulative_input_count = 0
        for input_length in sorted(input_length_counter.keys(), reverse=True):
            count = input_length_counter[input_length]
            accumulative_input_count += count
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count,
                                                                                      100.0 * count / total_number_of_tasks, 100.0 * accumulative_input_count / total_number_of_tasks))
        print("")

        print("How often output lengths occur:")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys()):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count,
                                                                                          100.0 * count / total_number_of_tasks, 100.0 * accumulative_output_count / total_number_of_tasks))
        print("")

        print("How often output lengths occur (reversed):")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys(), reverse=True):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count,
                                                                                          100.0 * count / total_number_of_tasks, 100.0 * accumulative_output_count / total_number_of_tasks))
        print("")


def create_split(max_train_length):
    input_file_all = open('tasks.txt', 'r')
    output_path = os.path.join('length_split', 'increasing_lengths', str(max_train_length))

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file_train = open(os.path.join(output_path, 'tasks_train.txt'), 'w')
    output_file_dev   = open(os.path.join(output_path, 'tasks_dev.txt'), 'w')
    output_file_test  = open(os.path.join(output_path, 'tasks_test.txt'), 'w')

    n_included_train = 0
    n_included_dev = 0
    n_included_test = 0

    actual_included = set()
    actual_excluded = set()

    for line in input_file_all:
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        output_sequence_length = len(output_sequence.split())

        if output_sequence_length <= max_train_length:
            if numpy.random.random() < 0.1:
                output_file_dev.write(line)
                n_included_dev += 1
                actual_included.add(output_sequence_length)
            else:
                output_file_train.write(line)
                n_included_train += 1
                actual_included.add(output_sequence_length)
        else:
            output_file_test.write(line)
            n_included_test += 1
            actual_excluded.add(output_sequence_length)

    total_number_of_tasks = n_included_train + n_included_dev + n_included_test

    print("Included output lengths in training set ({:.02f}%): {}".format(100.0 * n_included_train / total_number_of_tasks, sorted(actual_included)))
    print("Included output lengths in dev set ({:.02f}%): {}".format(100.0 * n_included_dev / total_number_of_tasks, sorted(actual_included)))
    print("Included output lengths in test set ({:.02f}%): {}".format(100.0 * n_included_test / total_number_of_tasks, sorted(actual_excluded)))
    print("Results are in '{}'".format(output_path))
    print()

if __name__ == '__main__':
    show_statistics()

    for max_train_length in [16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]:
        create_split(max_train_length)
