from seq2seq.util.log import LogCollection
import re
from collections import OrderedDict

def name_parser(filename, subdir):
    return filename.split('/')[-1]
    # if '64' in subdir:
    #     return splits[1][:index] + '_E64' + splits[1][index:]
    # else:
    #     return splits[1][:index] + '_E32' + splits[1][index:]

log = LogCollection()
# log.add_log_from_folder('dumps', ext='LOG', name_parser=name_parser)
# log.add_log_from_folder('dumps_64', ext='LOG', name_parser=name_parser)
# log.add_log_from_folder('dump_final', ext='LOG', name_parser=name_parser)
# log.add_log_from_folder('dumps_temp', ext='LOG', name_parser=name_parser)
log.add_log_from_folder('temp', ext='LOG', name_parser=name_parser)

############################
# helper funcs

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def data_name_parser(data_name, model_name):
    if 'baseline_E3' in model_name:
        label = 'Baseline (32x256)'
    elif 'baseline_E6' in model_name:
        label = 'Baseline (32x256)'
    elif 'hard' in model_name:
        label = 'Guided'
    else:
        label = 'Learned'
    label=''

    if 'basel' in model_name:
        label = "Baseline"
    else:
        label = "Seq2Attn"

    if 'Train' in data_name:
        label += ', Train'
    elif 'test' in data_name:
        label += ', Validation'
    print(label)

    return label

def color_groups(model_name, data_name):
    if 'train' in data_name:
        c = 'black'
    else:
        c = 'green'

    if 'baseline' in model_name:
        l = '-'
    elif 'learned' in model_name:
        l='--'
    else:
        l =':'

    return c,l


def basename_without_run(model_name):
    all_parts = model_name.split('_')
    basename = '_'.join(all_parts[:-1])
    return basename
def basename_with_run(model_name):
    all_parts = model_name.split('_')
    basename = '_'.join(all_parts[:])
    return basename

def no_restriction(input_str):
    return True
def k_base_name(input_str):
    return "_".join(input_str.split("_")[:-2])

def k_parse_data_set_name(dataset):
    dataname = dataset.split('/')[-1].split('.')[0]
    return dataname
    if 'long' in dataname:
        return 'long'
    elif 'short' in dataname:
        return 'short'
    if 'repeat' in dataname:
        return 'repeat'
    if 'Train' in dataname:
        return 'train'
    if 'std' in dataname:
        return 'standard'
    return 'validation'

def group_rest(dataset):
    dataname = dataset.split('/')[-1].split('.')[0]
    if 'Train' in dataname:
        return 'train'
    else:
        return 'tests'

mean_avg, min_avg, max_avg, std_avg = log.find_highest_average_val('rain', 'bleu', 'bleu', find_basename=k_base_name, find_data_name=k_parse_data_set_name, restrict_data=no_restriction)

print("\nData sets:")
for model in natural_sort(min_avg)[:1]:
    datadict = OrderedDict(sorted(min_avg[model].items(), key=lambda x : x[0], reverse=False))
    print('%s\t%s' % (".", '\t'.join(['%s' % (d) for d in datadict])))
    # print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))

print("\nMin:")
for model in natural_sort(min_avg):
    datadict = OrderedDict(sorted(min_avg[model].items(), key=lambda x : x[0], reverse=False))
    # print('%s:\t%s' % (model, '\t'.join(['%s %.4f' % (d, datadict[d]) for d in datadict])))
    print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))

print("\nMax:")
for model in natural_sort(max_avg):
    datadict = OrderedDict(sorted(max_avg[model].items(), key=lambda x : x[0], reverse=False))
    # print('%s:\t%s' % (model, '\t'.join(['%s %.4f' % (d, datadict[d]) for d in datadict])))
    print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))

print("\nMean:")
for i, model in enumerate(natural_sort(mean_avg)):
    datadict = OrderedDict(sorted(mean_avg[model].items(), key=lambda x : x[0], reverse=False))
    # print('%s:\t%s' % (model, '\t'.join(['%s %.4f' % (d, datadict[d]) for d in datadict])))
    print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))

print("\nStd:")
for model in natural_sort(std_avg):
    datadict = OrderedDict(sorted(std_avg[model].items(), key=lambda x : x[0], reverse=False))
    # print('%s:\t%s' % (model, '\t'.join(['%s %.4f' % (d, datadict[d]) for d in datadict])))
    print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))

print("\nMean+std:")
for i, model in enumerate(natural_sort(mean_avg)):
    datadict = OrderedDict(sorted(mean_avg[model].items(), key=lambda x : x[0], reverse=False))
    datadict_std = OrderedDict(sorted(std_avg[model].items(), key=lambda x: x[0], reverse=False))
    # print('%s:\t%s' % (model, '\t'.join(['%s %.4f' % (d, datadict[d]) for d in datadict])))
    # print('%s:\t%s' % (model, '\t'.join(['%.4f' % (datadict[d]) for d in datadict])))
    print('%s:\t%s' % (model, '\t'.join(['%.4f ($\\pm$%.2f)' % (datadict[d], datadict_std[d]) for d in datadict])))

def k_restrict(input_str):
    return True
    if 'learned' in input_str:
        return True
    if 'learned_E32_H256_SCALE1_run_' in input_str:
        return False
    if 'baseline_E64_H64_run_' in input_str:
        return False
    if 'baseline' in input_str:
        return False
    return False
def only_train(input_str):
    if 'Train' in input_str:
        return True
    return False
def only_validation(input_str):
    if 'test' in input_str:
        return True
    return False
def no_validation(input_str):
    if 'val' not in input_str and 'Train' not in input_str:
        return True
    return False
def only_long(input_str):
    if 'long' in input_str:
        return True
    return False
def train_and_val(input_str):
    return only_train(input_str) or only_validation(input_str)

def k_best_model_filter(input_str):
    if 'seq2attn_s' in input_str or 'baseline_f' in input_str:
        return True
    return False
    if 'full' not in input_str and 'hard' not in input_str:
        return True
    return False

def k_color_one(model_name, data_name):
    if 'baseline_E6' in model_name:
        c = 'black'
    elif 'baseline_E3' in model_name:
        c = 'm'
    elif 'hard' in model_name:
        c = 'g'
    elif 'learned' in model_name:
        c = 'b'

    if 'Train' in data_name:
        l = '-'
    else:
        l = '--'

    return c,l

def k_color_one_new(model_name, data_name):
    if 'baseline_full' in model_name:
        c = 'm'
    elif 'baseline_spa' in model_name:
        c = 'm'
    elif 'seq2attn_f' in model_name:
        c = 'g'
    elif 'seq2attn_s' in model_name:
        c = 'blue'

    if 'Train' in data_name:
        l = '--'
    else:
        l = '-'

    return c, l

# # Train loss
fig = log.plot_metric('bleu', restrict_model=k_best_model_filter, restrict_data=train_and_val, data_name_parser=data_name_parser, color_group=k_color_one_new, eor=-1, ylabel='BLEU')
fig.savefig('/home/kris/Desktop/Results plots/train_and_val_loss.png')

# # Train accuracy
# fig = log.plot_metric('sym_rwr_acc', restrict_model=k_best_model_filter, restrict_data=only_train, data_name_parser=data_name_parser, color_group=k_color_one_new, eor=-1, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/train_acc.png')

# # Validation accuracy
# fig = log.plot_metric('sym_rwr_acc', restrict_model=k_best_model_filter, restrict_data=only_validation, data_name_parser=data_name_parser, color_group=k_color_one_new, eor=-1, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/validation_acc.png')

# # Long accuracy
# fig = log.plot_metric('sym_rwr_acc', restrict_model=k_best_model_filter, restrict_data=only_long, data_name_parser=data_name_parser, color_group=k_color_one_new, eor=-1, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/longer_acc.png')

def second_parser(model_name, data_name):
    m = None
    if 'baseline' in model_name and 'with_' in model_name:
        m = 'Baseline with EOS'
    if 'baseline' in model_name and 'without' in model_name:
        m = 'Baseline (64x64)'
    if 'learned' in model_name:
        m = 'Guided'
    if 'hard' in model_name:
        m = 'Hard guidance'

    if 'long' in data_name:
        d = 'Long'
    elif 'short' in data_name:
        d = 'Short'
    elif 'repeat' in data_name:
        d = 'Repeat'
    elif 'train' in data_name:
        d = 'Train'
    elif 'standard' in data_name:
        d = 'Standard'
    elif 'val' in data_name:
        d = 'Validation'
    else:
        d = 'Tests combined'

    # return m
    return m + ', ' + d

# fig = log.plot_groups('sym_rwr_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=group_rest, restrict_data=no_validation, color_group=color_groups, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_combined_tests.png')
# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=group_rest, restrict_data=no_validation, color_group=color_groups, ylabel='Loss')
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_combined_tests.png')

# fig = log.plot_groups('sym_rwr_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=no_validation, color_group=color_groups)
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_individual_tests.png')
# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=no_validation, color_group=color_groups)
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_individual_tests.png')

def train_and_standard(input_str):
    if 'Train' in input_str or 'std' in input_str:
        return True
    return False
def train_and_repeat(input_str):
    if 'Train' in input_str or 'repeat' in input_str:
        return True
    return False
def train_and_short(input_str):
    if 'Train' in input_str or 'short' in input_str:
        return True
    return False
def train_and_long(input_str):
    if 'Train' in input_str or 'long' in input_str:
        return True
    return False

def only_repeat(input_str):
    return 'repeat' in input_str
def only_standard(input_str):
    return 'std' in input_str
def only_short(input_str):
    return 'short' in input_str
def only_long(input_str):
    return 'long' in input_str

def color_groups2(model_name, data_name):
    if 'train' in data_name:
        c = 'black'
        l = '--'
    elif '64' in model_name:
        c = 'm'
        l = '-'
    elif '32' in model_name:
        c = 'b'
        l = '-'
    elif 'learned' in model_name:
        l = '-'
        c = 'g'
    else:
        l = '-'
        c = 'black'

    return c,l

# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_standard, color_group=color_groups2, ylabel='Loss')
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_standard.png')
# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_repeat, color_group=color_groups2, ylabel='Loss')
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_repeat.png')
# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_short, color_group=color_groups2, ylabel='Loss')
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_short.png')
# fig = log.plot_groups('nll_loss', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_long, color_group=color_groups2, ylabel='Loss')
# fig.savefig('/home/kris/Desktop/Results plots/average_loss_train_long.png')

# fig = log.plot_groups('k_grammar_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_standard, color_group=color_groups2, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_standard.png')
# fig = log.plot_groups('k_grammar_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_repeat, color_group=color_groups2, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_repeat.png')
# fig = log.plot_groups('k_grammar_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_short, color_group=color_groups2, ylabel='Accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_short.png')
# fig = log.plot_groups('k_grammar_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=k_parse_data_set_name, restrict_data=only_long, color_group=color_groups2, ylabel='accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average_acc_train_long.png')

# fig = log.plot_groups('k_grammar_acc', restrict_model=k_best_model_filter, find_basename=basename_without_run, data_name_parser=second_parser, find_data_name=group_rest, restrict_data=no_validation, color_group=color_groups2, ylabel='accuracy')
# fig.savefig('/home/kris/Desktop/Results plots/average.png')
