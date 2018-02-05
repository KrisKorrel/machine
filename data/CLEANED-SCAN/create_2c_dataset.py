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


def create_split(split_on, included):
    """
    Creates a train/test split on basis of either input or output length of
    the examples

    Args:
        split_on (str): Split on either 'input' or 'output'
        included (list): A list of all the lengths that should be included in the
        split_name (str): Name of the directory in which the train/test files will be stored
        training set. The rest are in the test set
    """
    input_file_all = open('tasks.txt', 'r')

    if not os.path.exists(os.path.join('length_split', '2c-train-dev-test')):
        os.mkdir(os.path.join('length_split', '2c-train-dev-test'))
    
    output_file_train = open(os.path.join(
        'length_split', '2c-train-dev-test', 'tasks_train.txt'), 'w')
    output_file_dev = open(os.path.join(
        'length_split', '2c-train-dev-test', 'tasks_dev.txt'), 'w')
    output_file_test = open(os.path.join(
        'length_split', '2c-train-dev-test', 'tasks_test.txt'), 'w')

    n_included_train = 0
    n_included_dev = 0
    n_included_test = 0

    actual_included_train = set()
    actual_included_dev = set()
    actual_included_test = set()

    for line in input_file_all:
        line_stripped = line.strip()
        input_sequence, output_sequence = line_stripped.split('\t')

        input_sequence_length = len(input_sequence.split())
        output_sequence_length = len(output_sequence.split())

        if split_on == 'input':
            sequence_length_to_split_on = input_sequence_length
        elif split_on == 'output':
            sequence_length_to_split_on = output_sequence_length
        else:
            print("Incorrect argument")

        if sequence_length_to_split_on in included:
            if numpy.random.random() < 0.1:
                output_file_dev.write(line)
                n_included_dev += 1
                actual_included_dev.add(sequence_length_to_split_on)
            else:
                output_file_train.write(line)
                n_included_train += 1
                actual_included_train.add(sequence_length_to_split_on)
        else:
            output_file_test.write(line)
            n_included_test += 1
            actual_included_test.add(sequence_length_to_split_on)

    total_number_of_tasks = n_included_train + n_included_dev + n_included_test

    print("Included {} lengths in training set ({:.02f}%): {}".format(split_on, 100.0 * n_included_train / total_number_of_tasks, sorted(actual_included_train)))
    print("Included {} lengths in dev set ({:.02f}%): {}".format(split_on, 100.0 * n_included_dev / total_number_of_tasks, sorted(actual_included_dev)))
    print("Included {} lengths in test set ({:.02f}%): {}".format(split_on, 100.0 * n_included_test / total_number_of_tasks, sorted(actual_included_test)))
    print()

if __name__ == '__main__':
    show_statistics()

    create_split(split_on='output', included=[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22])