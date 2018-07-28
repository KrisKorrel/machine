from __future__ import division
import logging
import os
import random
import time
import shutil

import torch
import torchtext
from torch import optim

from collections import defaultdict

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss, AttentionLoss
from seq2seq.metrics import SymbolRewritingAccuracy
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.log import Log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (list, optional): list of seq2seq.loss.Loss objects for training (default: [seq2seq.loss.NLLLoss])
        metrics (list, optional): list of seq2seq.metric.metric objects to be computed during evaluation
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        print_every (int, optional): number of iterations to print after, (default: 100)
    """
    def __init__(self, understander_train_method, train_regime, expt_dir='experiment', loss=[NLLLoss()], loss_weights=None, metrics=[], batch_size=64, eval_batch_size=128,
                 random_seed=None,
                 checkpoint_every=100, print_every=100, epsilon=1):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        k = NLLLoss()
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights or len(loss)*[1.]
        self.evaluator = Evaluator(loss=self.loss, metrics=self.metrics, batch_size=eval_batch_size)
        self.optimizer = None
        self.total_optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        self.epsilon = epsilon
        self.understander_train_method = understander_train_method
        self.train_regime = train_regime

        # Start either in 'pre-train' or 'train' mode
        if self.train_regime == 'two-stage':
            self.pre_train = True
        else:
            self.pre_train = False

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss

        decoder_outputs, decoder_hidden, other = model.forward(
            input_variable=input_variable,
            input_lengths=input_lengths.tolist(),
            target_variables=target_variable,
            teacher_forcing_ratio=teacher_forcing_ratio)

        model.zero_grad()
        
        # Calculate the losses of the executor
        losses = self.evaluator.compute_batch_loss(decoder_outputs, decoder_hidden, other, target_variable)

        if self.understander_train_method == 'rl':
            # TODO: This loss metric should be initialized in train_model and passed to the trainer. (And we shouldn't hard-code the ignore_index value)
            # Actually, I think we should use word *accuracy* instead of loss for the reward function of the understander.
            # At least, we shouldn't rewrite code that is already in loss.py and metrics.py
            loss_func = torch.nn.NLLLoss(ignore_index=-1, reduce=False)

            rewards = []
            for action_iter in range(len(decoder_outputs)):
                prediction = decoder_outputs[action_iter]

                # +1 because target_variable includes SOS which the prediction of course doesn't
                ground_truth = target_variable['decoder_output'][:,action_iter+1]
                
                # Since loss is usually in range [0-3], where a loss of 0 should give the highest reward to the undestander,
                # we use as reward function: (1/3) * (3-loss).
                # This is because RL seems to be very unstable when we allow negative rewards. We even clip the rewards
                # to [0,1] to make sure this doesn't happen
                import numpy
                step_reward = list(numpy.clip((3-loss_func(prediction, ground_truth).detach().cpu().numpy())/3, 0, 1))
                rewards.append(step_reward)

            model.decoder.decoder_model.set_rewards(rewards)

            # Calculate discounted rewards and policy loss
            policy_loss = model.decoder.decoder_model.finish_episode()
            policy_loss.backward(retain_graph=True)

        for i, loss in enumerate(losses):
            loss.scale_loss(self.loss_weights[i])
            loss.backward(retain_graph=True)
        self.optimizer.step()

        return losses

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, pre_train=None,
                       dev_data=None, monitor_data=[], teacher_forcing_ratio=0, top_k=5):
        log = self.logger

        print_loss_total = defaultdict(float)  # Reset every print_every
        epoch_loss_total = defaultdict(float)  # Reset every epoch
        epoch_loss_avg = defaultdict(float)
        print_loss_avg = defaultdict(float)

        iterator_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        batch_iterator_train = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=iterator_device, repeat=False)

        batch_iterator_pre_train = torchtext.data.BucketIterator(
            dataset=pre_train, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=iterator_device, repeat=False)

        steps_per_epoch = len(batch_iterator_train)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        # store initial model to be sure at least one model is stored
        val_data = dev_data or data
        losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)

        total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)
        log.info(log_msg)

        # For the SR task we look at SR accuracy instead of validation loss.
        # We take the negative, as the checks still work then. Kinda hacky..
        for metric in metrics:
            if isinstance(metric, SymbolRewritingAccuracy):
                total_loss = -1 * metric.get_val()
                print(total_loss)

        logs = Log()
        loss_best = top_k*[total_loss]
        best_checkpoints = top_k*[None]
        best_checkpoints[0] = model_name

        Checkpoint(model=model,
                   optimizer=self.optimizer,
                   epoch=start_epoch, step=start_step,
                   input_vocab=data.fields[seq2seq.src_field_name].vocab,
                   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

        if self.pre_train:
            # Disable training updates for the understander
            model.train_understander(train=False)
            model.train_executor(train=True)

        for epoch in range(start_epoch, n_epochs + 1):
            try:
                if model.decoder.decoder_model.attention.current_temperature is not None:
                    log.info("Example temperature: {}".format(model.decoder.decoder_model.attention.current_temperature[0].item()))
            except Exception:
                pass

            log.info("Epoch: %d, Step: %d" % (epoch, step))

            if self.train_regime == 'two-stage':
                # First 50% of epochs we are in pre-train. The next we are in train mode
                if epoch < n_epochs / 2:
                    self.pre_train = True
                    batch_generator = batch_iterator_pre_train.__iter__()

                else:
                    if self.pre_train:
                        raw_input("Pre-training is done. Press enter to start training the undestander")
                        # Disable training updates for the executor
                        model.train_understander(train=True)
                        model.train_executor(train=False)
                        
                    self.pre_train = False
                    batch_generator = batch_iterator_train.__iter__()

            elif self.train_regime == 'simultaneous':
                # TODO: What to do about this?
                # consuming seen batches from previous training
                batch_generator = batch_iterator_train.__iter__()

                for _ in range((epoch - 1) * steps_per_epoch, step):
                    next(batch_generator)

                # Makes sure that we use the understander's output in the Evaluator instead of the data
                self.pre_train = False

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths, target_variables = self.get_batch_data(batch)

                losses = self._train_batch(input_variables, input_lengths, target_variables, model, teacher_forcing_ratio)

                # Record average loss
                for loss in losses:
                    name = loss.log_name
                    print_loss_total[name] += loss.get_loss()
                    epoch_loss_total[name] += loss.get_loss()

                # print log info according to print_every parm
                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    for loss in losses:
                        name = loss.log_name
                        print_loss_avg[name] = print_loss_total[name] / self.print_every
                        print_loss_total[name] = 0

                    m_logs = {}
                    train_losses, train_metrics = self.evaluator.evaluate(model, data, self.get_batch_data)
                    train_loss, train_log_msg, model_name = self.get_losses(train_losses, train_metrics, step)
                    logs.write_to_log('Train', train_losses, train_metrics, step)
                    logs.update_step(step)

                    m_logs['Train'] = train_log_msg

                    # compute vals for all monitored sets
                    for m_data in monitor_data:
                        losses, metrics = self.evaluator.evaluate(model, monitor_data[m_data], self.get_batch_data)
                        total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)
                        m_logs[m_data] = log_msg
                        logs.write_to_log(m_data, losses, metrics, step)

                    all_losses = ' '.join(['%s:\t %s\n' % (os.path.basename(name), m_logs[name]) for name in m_logs])

                    log_msg = 'Progress %d%%\n%s' % (
                            step / total_steps * 100,
                            all_losses)

                    log.info(log_msg)

                # check if new model should be saved
                if step % self.checkpoint_every == 0 or step == total_steps:
                    # compute dev loss
                    losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)
                    total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)

                    # For the SR task we look at SR accuracy instead of validation loss.
                    # We take the negative, as the checks still work then. Kinda hacky..
                    for metric in metrics:
                        if isinstance(metric, SymbolRewritingAccuracy):
                            total_loss = -1 * metric.get_val()
                    max_eval_loss = max(loss_best)

                    if total_loss < max_eval_loss:
                            index_max = loss_best.index(max_eval_loss)
                            # rm prev model
                            if best_checkpoints[index_max] is not None:
                                shutil.rmtree(os.path.join(self.expt_dir, best_checkpoints[index_max]))
                            best_checkpoints[index_max] = model_name
                            loss_best[index_max] = total_loss

                            # save model
                            Checkpoint(model=model,
                                       optimizer=self.optimizer,
                                       epoch=epoch, step=step,
                                       input_vocab=data.fields[seq2seq.src_field_name].vocab,
                                       output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

            if step_elapsed == 0: continue

            for loss in losses:
                epoch_loss_avg[loss.log_name] = epoch_loss_total[loss.log_name] / min(steps_per_epoch, step - start_step)
                epoch_loss_total[loss.log_name] = 0

            loss_msg = ' '.join(['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in losses])
            log_msg = "Finished epoch %d: Train %s" % (epoch, loss_msg)

            if dev_data is not None:
                losses, metrics = self.evaluator.evaluate(model, dev_data, self.get_batch_data)
                loss_total, log_, model_name = self.get_losses(losses, metrics, step)

                self.optimizer.update(loss_total, epoch)    # TODO check if this makes sense!
                log_msg += ", Dev set: " + log_


                losses, metrics = self.evaluator.evaluate(model, data, self.get_batch_data)
                loss_total, log_, model_name = self.get_losses(losses, metrics, step)

                log_msg += ", Train set: " + log_

                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch) # TODO check if this makes sense!

            log.info(log_msg)

        return logs

    def train(self, model, data, pre_train=None, num_epochs=5,
              resume=False, dev_data=None, optimizer=None,
              teacher_forcing_ratio=0, monitor_data={},
              learning_rate=0.001, checkpoint_path=None, top_k=5):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0

            def get_optim(optim_name):
                optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
                          'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
                          'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
                           None:optim.Adam}
                return optims[optim_name]

            self.optimizer = Optimizer(get_optim(optimizer)(model.parameters(), lr=learning_rate),max_grad_norm=5)
        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        logs = self._train_epoches(data, model, num_epochs,
                            start_epoch, step, pre_train=pre_train, dev_data=dev_data,
                            monitor_data=monitor_data,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            top_k=top_k)

        return model, logs

    @staticmethod
    def get_batch_data(batch):
        input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
        target_variables = {'decoder_output': getattr(batch, seq2seq.tgt_field_name),
                            'encoder_input': input_variables}  # The k-grammar metric needs to have access to the inputs

        # If available, also get provided attentive guidance data
        if hasattr(batch, seq2seq.attn_field_name):
            attention_target = getattr(batch, seq2seq.attn_field_name)
            target_variables['attention_target'] = attention_target

        return input_variables, input_lengths, target_variables

    @staticmethod
    def get_losses(losses, metrics, step):
        total_loss = 0
        model_name = ''
        log_msg= ''

        for metric in metrics:
            val = metric.get_val()
            log_msg += '%s %.4f ' % (metric.name, val)
            model_name += '%s_%.2f_' % (metric.log_name, val)

        for loss in losses:
            val = loss.get_loss()
            log_msg += '%s %.4f ' % (loss.name, val)
            model_name += '%s_%.2f_' % (loss.log_name, val)
            total_loss += val

        model_name += 's%d' % step

        return total_loss, log_msg, model_name
