from collections import Counter

def show_statistics():
    input_length_counter = Counter()
    output_length_counter = Counter()

    with open('tasks.txt') as all_tasks_file:
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
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count, 100.0*count/total_number_of_tasks, 100.0*accumulative_input_count/total_number_of_tasks))
        print("")

        print("How often input lengths occur (reversed):")
        accumulative_input_count = 0
        for input_length in sorted(input_length_counter.keys(), reverse=True):
            count = input_length_counter[input_length]
            accumulative_input_count += count
            print("{} occurs {: <4} times ({:5.02f}%). Accumulative: {:.02f}%".format(input_length, count, 100.0*count/total_number_of_tasks, 100.0*accumulative_input_count/total_number_of_tasks))
        print("")

        print("How often output lengths occur:")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys()):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count, 100.0*count/total_number_of_tasks, 100.0*accumulative_output_count/total_number_of_tasks))
        print("")

        print("How often output lengths occur (reversed):")
        accumulative_output_count = 0
        for output_length in sorted(output_length_counter.keys(), reverse=True):
            count = output_length_counter[output_length]
            accumulative_output_count += count
            print("{: <2} occurs {: <4} times ({:4.02f}%). Accumulative: {:.02f}%".format(output_length, count, 100.0*count/total_number_of_tasks, 100.0*accumulative_output_count/total_number_of_tasks))
        print("")

if __name__ == '__main__':
    show_statistics()