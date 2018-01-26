import re
import matplotlib.pyplot as plt


def split_results(filename):
    """
    To fully utilize the cartesius nodes, we run two experiments on one node.
    This means that the output of both experiments are written to the same file.
    This function reads in that file and outputs to two different files.
    
    Args:
        filename (str): Filename of the input file
    """
    f_in = open(filename, 'r')
    f_out_1 = open(filename + '-0', 'w')
    f_out_2 = open(filename + '-1', 'w')

    for line in f_in:
        # Specific for task 1
        if line[0] == '0':
            f_out_1.write(line[3:])
        # Specific for task 2
        elif line[0] == '1':
            f_out_2.write(line[3:])
        # Common
        else:
            f_out_1.write(line)
            f_out_2.write(line)

    f_in.close()
    f_out_1.close()
    f_out_2.close()


def plot_results(filename):
    """
    Parse the output file and plot the perplexity and accuracy after every full epoch
    
    Args:
        filename (str): filename of the output file of a run
    """
    train_losses = []
    dev_losses = []
    dev_word_accuracies = []
    dev_sequence_accuracies = []

    perce = 0

    with open(filename) as f_in:
        for line in f_in:
            re_train_loss_results = re.findall(r'Train Perplexity: ([0-9]+\.[0-9]+)', line)
            re_dev_loss_results = re.findall(r'Dev Perplexity: ([0-9]+\.[0-9]+)', line)
            re_dev_word_acc_results = re.findall(r', Accuracy: ([0-9]+\.[0-9]+)', line)
            re_dev_seq_acc_results = re.findall(r'Sequence Accuracy: ([0-9]+\.[0-9]+)', line)

            perc = re.findall(r'Included lengths in training set \((.*)%\)', line)

            if len(perc) > 0:
                perce = perc[0]

            if len(re_dev_loss_results) > 0:
                train_losses += re_train_loss_results
                dev_losses += re_dev_loss_results
                dev_word_accuracies += re_dev_word_acc_results
                dev_sequence_accuracies += re_dev_seq_acc_results

    n_epochs = len(train_losses)

    # plt.plot(range(1, n_epochs + 1), train_losses, label='Train perplexity')
    # plt.plot(range(1, n_epochs + 1), dev_losses, label='Dev perplexity')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.xlabel('Epoch')
    # plt.ylabel('Perplexity')
    # plt.savefig(filename + '-loss')

    # plt.clf()

    # plt.plot(range(1, n_epochs + 1), dev_word_accuracies, label='Dev word accuracy')
    plt.plot(range(1, n_epochs + 1), dev_sequence_accuracies, label='Dev sequence accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0,0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Áccuracy')
    plt.savefig(filename + '-accuracy')

    plt.clf()

    print("Max sequence accuracy {}: {} (train set size: {}%)".format(filename, max(dev_sequence_accuracies), float(perce)))

    return float(max(dev_sequence_accuracies))

if __name__ == '__main__':
    filenames = ['1', '2a', '2b', '2c', '2d', '2e', '2f']
    filenames = ['exp' + f for f in filenames]
    filenames = ['exp_random_leave_out_{}'.format(i) for i in range(5)]

    for filename in filenames:
        # split_results(filename)
        max1 = plot_results(filename + '-0')
        max2 = plot_results(filename + '-1')
        max3 = plot_results(filename + '-2')
        max4 = plot_results(filename + '-3')

        print("Average max accuracy: {}\n".format((max1 + max2 + max3 + max4) / 4))

    filenames = ['exp_random_leave_out_{}'.format(i) for i in range(5,12)]

    for filename in filenames:
        # split_results(filename)
        max1 = plot_results(filename + '-0')
        max2 = plot_results(filename + '-1')
        max3 = plot_results(filename + '-2')
        max4 = plot_results(filename + '-3')
        max5 = plot_results(filename + '-4')
        max6 = plot_results(filename + '-5')

        print("Average max accuracy: {}\n".format((max1 + max2 + max3 + max4 + max5 + max6) / 6))
