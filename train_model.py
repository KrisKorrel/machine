import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import random
import pickle

from collections import OrderedDict

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, Understander
from seq2seq.loss import Perplexity, AttentionLoss, NLLLoss, PonderLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy, SymbolRewritingAccuracy
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField, AttentionField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training data')
parser.add_argument('--dev', help='Development data')
parser.add_argument('--monitor', nargs='+', default=[], help='Data to monitor during training')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--optim', type=str, help='Choose optimizer', choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--rnn_cell', type=str, help="Chose type of rnn cell", choices=['lstm', 'gru'], default='lstm')
parser.add_argument('--bidirectional', action='store_true', help="Flag for bidirectional encoder")
parser.add_argument('--embedding_size', type=int, help='Embedding size', default=128)
parser.add_argument('--hidden_size', type=int, help='Hidden layer size', default=128)
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder', default=1)
parser.add_argument('--src_vocab', type=int, help='source vocabulary size', default=50000)
parser.add_argument('--tgt_vocab', type=int, help='target vocabulary size', default=50000)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder', default=0.2)
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the decoder', default=0.2)
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn', 'seq2attn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'concat', 'hard'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--scale_attention_loss', type=float, default=1.)
parser.add_argument('--xent_loss', type=float, default=1.)
parser.add_argument('--metrics', nargs='+', default=['seq_acc'], choices=['word_acc', 'seq_acc', 'target_acc', 'sym_rwr_acc'], help='Metrics to use')
parser.add_argument('--full_focus', action='store_true')
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--eval_batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

parser.add_argument('--load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--save_every', type=int, help='Every how many batches the model should be saved', default=100)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=100)
parser.add_argument('--resume', action='store_true', help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', default='info', help='Logging level.')
parser.add_argument('--write-logs', help='Specify file to write logs to after training')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

# Arguments for the UE model
parser.add_argument('--pre_train', help='Data for pre-training the executor')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma to use for discounted future rewards')
parser.add_argument('--epsilon', type=float, default=1, help='Epsilon to use for epsilon-greedy during RL training.')
parser.add_argument('--understander_train_method', type=str, choices=['rl', 'supervised'], help='Whether to use reinforcement or supervised learning for the understander')
parser.add_argument('--sample_train', type=str, choices=['full', 'gumbel_soft', 'gumbel_hard'], help='When training UE in a supervised setting, we can use the full attention vector or sample using gumbel (ST) at training time')
parser.add_argument('--sample_infer', type=str, choices=['full', 'gumbel_soft', 'gumbel_hard', 'argmax'], help='When training UE in a supervised setting, we can use the full attention vector, sample using gumbel (ST), or use argmax at inference time')
parser.add_argument('--initial_temperature', type=float, default=1, help='(Initial) temperature to use for gumbel-softmax')
parser.add_argument('--learn_temperature', type=str, choices=['no', 'unconditioned', 'conditioned'], help='Whether the temperature should be a learnable parameter. And whether it should be conditioned')
parser.add_argument('--init_exec_dec_with', type=str, choices=['encoder', 'new'], default='encoder', help='The decoder of the executor can be initialized either with its last encoder state, or with a new (learnable) vector')
# TODO: two-stage currently doesn't work anymore
# parser.add_argument('--train_regime', type=str, choices=['two-stage', 'simultaneous'], help="In 'two-stage' training we first train the executor with hard guidance for n/2 epochs and then the understander for n/2 epochs. In 'simultaneous' training, we train both models together without any supervision on the attention.")
parser.add_argument('--attn_keys', type=str, choices=['understander_encoder_embeddings', 'understander_encoder_outputs', 'executor_encoder_embeddings', 'executor_encoder_outputs'], default='executor_encoder_outputs')
parser.add_argument('--attn_vals', type=str, choices=['understander_encoder_embeddings', 'understander_encoder_outputs', 'executor_encoder_embeddings', 'executor_encoder_outputs'], default='executor_encoder_outputs')
parser.add_argument('--dropout_enc_dec', default='0', type=float, help="If the executor decoder is initialized with it's last encoder state, you can optionally add dropout between the encoder and decoder")

# Ponder arguments
parser.add_argument('--ponder_encoder', action='store_true', help='Use ACT pondering for the encoder')
parser.add_argument('--ponder_decoder', action='store_true', help='Use ACT pondering for the decoder')
parser.add_argument('--max_ponder_steps', type=int, default=100, help='Hard maximum number of ponder steps')
parser.add_argument('--ponder_epsilon', type=float, default=0.01, help='Epsilon for ACT to allow only 1 ponder step')
parser.add_argument('--ponder_penalty_scale', type=float, default=0.01, help='Scale of the ponder penalty loss')

opt = parser.parse_args()
IGNORE_INDEX=-1
use_output_eos = not opt.ignore_output_eos

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)


if opt.resume and not opt.load_checkpoint:
    parser.error('load_checkpoint argument is required to resume training from checkpoint')

if opt.use_attention_loss and not opt.attention:
    parser.error('Specify attention type to use attention loss')

if not opt.attention and opt.attention_method:
    parser.error("Attention method provided, but attention is not turned on")

if opt.attention and not opt.attention_method:
    parser.error("Attention turned on, but no attention method provided")

if opt.use_attention_loss and opt.attention_method == 'hard':
    parser.error("Can't use attention loss in combination with non-differentiable hard attention method.")

if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

if opt.attention:
    if not opt.attention_method:
        logging.info("No attention method provided. Using DOT method.")
        opt.attention_method = 'dot'

############################################################################
# Prepare dataset
src = SourceField()
tgt = TargetField(include_eos=use_output_eos)

tabular_data_fields = [('src', src), ('tgt', tgt)]

if opt.use_attention_loss or opt.attention_method == 'hard':
    attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
    tabular_data_fields.append(('attn', attn))

max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=tabular_data_fields,
    filter_pred=len_filter
)

if opt.pre_train is not None:
  pre_train = torchtext.data.TabularDataset(
      path=opt.pre_train, format='tsv',
      fields=tabular_data_fields,
      filter_pred=len_filter
  )
else:
  pre_train = None

if opt.dev:
    dev = torchtext.data.TabularDataset(
        path=opt.dev, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )

else:
    dev = None

monitor_data = OrderedDict()
for dataset in opt.monitor:
    m = torchtext.data.TabularDataset(
        path=dataset, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter)
    monitor_data[dataset] = m

# When chosen to use attentive guidance, check whether the data is correct for the first
# example in the data set. We can assume that the other examples are then also correct.
if opt.use_attention_loss or opt.attention_method == 'hard':
    if len(train) > 0:
        if 'attn' not in vars(train[0]):
            raise Exception("AttentionField not found in train data")
        tgt_len = len(vars(train[0])['tgt']) - 1 # -1 for SOS
        attn_len = len(vars(train[0])['attn']) - 1 # -1 for preprended ignore_index
        if attn_len != tgt_len:
            raise Exception("Length of output sequence does not equal length of attention sequence in train data")

    if dev is not None and len(dev) > 0:
        if 'attn' not in vars(dev[0]):
            raise Exception("AttentionField not found in dev data")
        tgt_len = len(vars(dev[0])['tgt']) - 1 # -1 for SOS
        attn_len = len(vars(dev[0])['attn']) - 1 # -1 for preprended ignore_index
        if attn_len != tgt_len:
            raise Exception("Length of output sequence does not equal length of attention sequence in dev data.")

    for m in monitor_data.values():
        if len(m) > 0:
            if 'attn' not in vars(m[0]):
                raise Exception("AttentionField not found in monitor data")
            tgt_len = len(vars(m[0])['tgt']) - 1 # -1 for SOS
            attn_len = len(vars(m[0])['attn']) - 1 # -1 for preprended ignore_index
            if attn_len != tgt_len:
                raise Exception("Length of output sequence does not equal length of attention sequence in monitor data.")

#################################################################################
# prepare model

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model

    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab

    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

else:
    # build vocabulary
    src.build_vocab(train, max_size=opt.src_vocab)
    tgt.build_vocab(train, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size*2 if opt.bidirectional else hidden_size
    understander_encoder = EncoderRNN(len(src.vocab),
                                      max_len,
                                      hidden_size,
                                      opt.embedding_size,
                                      dropout_p=opt.dropout_p_encoder,
                                      n_layers=opt.n_layers,
                                      bidirectional=opt.bidirectional,
                                      rnn_cell=opt.rnn_cell,
                                      variable_lengths=True,
                                      ponder=opt.ponder_encoder,
                                      max_ponder_steps=opt.max_ponder_steps,
                                      ponder_epsilon=opt.ponder_epsilon,
                                      ponder_key='understander_encoder_ponder_penalty')
    executor_encoder = EncoderRNN(len(src.vocab),
                                  max_len,
                                  hidden_size,
                                  opt.embedding_size,
                                  dropout_p=opt.dropout_p_encoder,
                                  n_layers=opt.n_layers,
                                  bidirectional=opt.bidirectional,
                                  rnn_cell=opt.rnn_cell,
                                  variable_lengths=True,
                                  ponder=opt.ponder_encoder,
                                  max_ponder_steps=opt.max_ponder_steps,
                                  ponder_epsilon=opt.ponder_epsilon,
                                  ponder_key='executor_encoder_ponder_penalty')
    decoder = DecoderRNN(len(tgt.vocab), max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id,
                         sos_id=tgt.sos_id,
                         embedding_dim=opt.embedding_size,
                         train_method=opt.understander_train_method,
                         gamma=opt.gamma,
                         epsilon=opt.epsilon,
                         sample_train=opt.sample_train,
                         sample_infer=opt.sample_infer,
                         initial_temperature=opt.initial_temperature,
                         learn_temperature=opt.learn_temperature,
                         init_exec_dec_with=opt.init_exec_dec_with,
                         attn_keys=opt.attn_keys,
                         attn_vals=opt.attn_vals,
                         ponder=opt.ponder_decoder,
                         max_ponder_steps=opt.max_ponder_steps,
                         ponder_epsilon=opt.ponder_epsilon)
    seq2seq = Seq2seq(understander_encoder, executor_encoder, decoder, dropout_enc_dec=opt.dropout_enc_dec)
    seq2seq.to(device)

    for param in seq2seq.named_parameters():
        name, data = param[0], param[1].data
        if "halt_layer.bias" not in name and 'temperature' not in name:
            data.uniform_(-0.08, 0.08)

input_vocabulary = input_vocab.itos
output_vocabulary = output_vocab.itos

##############################################################################
# train model

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad)]
# loss_weights = [1.]
loss_weights = [float(opt.xent_loss)]

if opt.ponder_encoder:
    losses.append(PonderLoss(name="Understander encoder ponder penalty", log_name="understander_encoder_ponder_loss", identifier="understander_encoder_ponder_penalty"))
    loss_weights.append(opt.ponder_penalty_scale)

    losses.append(PonderLoss(name="Executor encoder ponder penalty", log_name="executor_encoder_ponder_loss", identifier="executor_encoder_ponder_penalty"))
    loss_weights.append(opt.ponder_penalty_scale)

if opt.ponder_decoder:
    losses.append(PonderLoss(name="Decoder ponder penalty", log_name="decoder_ponder_loss", identifier="decoder_ponder_penalty"))
    loss_weights.append(opt.ponder_penalty_scale)

if opt.use_attention_loss:
    losses.append(AttentionLoss(ignore_index=IGNORE_INDEX))
    loss_weights.append(opt.scale_attention_loss)

for loss in losses:
    loss.to(device)

metrics = []
if 'word_acc' in opt.metrics:
    metrics.append(WordAccuracy(ignore_index=pad))
if 'seq_acc' in opt.metrics:
    metrics.append(SequenceAccuracy(ignore_index=pad))
if 'target_acc' in opt.metrics:
    metrics.append(FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id))
if 'sym_rwr_acc' in opt.metrics:
    metrics.append(SymbolRewritingAccuracy(
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        use_output_eos=use_output_eos,
        input_pad_symbol=src.pad_token,
        output_sos_symbol=tgt.SYM_SOS,
        output_pad_symbol=tgt.pad_token,
        output_eos_symbol=tgt.SYM_EOS,
        output_unk_symbol=tgt.unk_token))

checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint) if opt.resume else None

# create trainer
t = SupervisedTrainer(loss=losses,
                      metrics=metrics, 
                      loss_weights=loss_weights,
                      batch_size=opt.batch_size,
                      eval_batch_size=opt.eval_batch_size,
                      checkpoint_every=opt.save_every,
                      print_every=opt.print_every,
                      expt_dir=opt.output_dir,
                      epsilon=opt.epsilon,
                      understander_train_method=opt.understander_train_method,
                      train_regime='simultaneous') # TODO: two-stage currently doesn't work. Do we still want it?

seq2seq, logs = t.train(model=seq2seq,
                    data=train,
                    dev_data=dev,
                    monitor_data=monitor_data,
                    pre_train=pre_train,
                    num_epochs=opt.epochs,
                    optimizer=opt.optim,
                    teacher_forcing_ratio=opt.teacher_forcing_ratio,
                    learning_rate=opt.lr,
                    resume=opt.resume,
                    checkpoint_path=checkpoint_path)

if opt.write_logs:
    output_path = os.path.join(opt.output_dir, opt.write_logs)
    logs.write_to_file(output_path)

# evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
# dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
