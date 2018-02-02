import numpy

all_in = open('tasks.txt', 'r')
train_out = open('simple_split/train-dev-test/train.txt', 'w')
dev_out = open('simple_split/train-dev-test/dev.txt', 'w')
test_out = open('simple_split/train-dev-test/test.txt', 'w')

# tasks.txt is already shuffled...
n_total = 20910
n_train = int(n_total * 0.7)
n_dev = int(n_total * 0.1)
n_test = int(n_total * 0.2)

n_actual_train = 0
n_actual_dev = 0
n_actual_test = 0

n = 0
for line in all_in:
    if n < n_train:
        train_out.write(line)
        n_actual_train += 1
    elif n < n_train + n_dev:
        dev_out.write(line)
        n_actual_dev += 1
    else:
        test_out.write(line)
        n_actual_test += 1
    n += 1

print("Examples in train set: {} ({}%)".format(n_actual_train, 100.0 * n_actual_train / n))
print("Examples in dev set: {} ({}%)".format(n_actual_dev, 100.0 * n_actual_dev / n))
print("Examples in test set: {} ({}%)".format(n_actual_test, 100.0 * n_actual_test / n))