import argparse
import torch
import datetime
import pathlib
import sys
import subprocess
import json
import numpy as np

from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dailydialogue import DailyDialogueDataset
from opensubtitles import OpenSubtitlesDataset
from dateutil import tz
from eval import eval_model
from util import build_dd_tests_from_csv
from pathlib import PosixPath


DEFAULT_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]


class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


class BaseTrainer():

    LOG_ROOT_DIR = 'log/'

    def __init__(self,
        model=None,
        train_dataset=None,
        eval_dataset=None,
        num_epochs=1000,
        learning_rate=5e-5,
        log_every=100,
        batch_size=64,
        save_models=True,
        log_root_dir=None,
        sanity=False,
        save_every=1,
        resume_path='',
    ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tzone = tz.gettz('America/Edmonton')
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

        if resume_path:

            self.model = model.cuda()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.scaler = torch.cuda.amp.GradScaler()

            ckpt = torch.load(PosixPath(resume_path, 'last_checkpoint.pt'))

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])

            with open(PosixPath(resume_path, 'last_state.json'), mode='r') as f:
                state = json.load(f)

            self.training_steps = state['training_steps']
            self.epoch = state['epoch'] + 1
            self.global_step = state['global_step']

            # logging
            self.log_root_dir = PosixPath(resume_path).parent
            self.log_dir = resume_path

        else:

            self.model = model
            # Set up for optimizer
            self.learning_rate = learning_rate
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scaler = torch.cuda.amp.GradScaler()

            self.training_steps = 0
            self.epoch = 1
            self.global_step = 0

            # Set up for logging
            if not log_root_dir:
                log_root_dir = BaseTrainer.LOG_ROOT_DIR
            self.log_root_dir = pathlib.PosixPath(log_root_dir)
            if not self.log_root_dir.exists():
                self.log_root_dir.mkdir()
            self.log_dir = pathlib.PosixPath(self.log_root_dir, self.timestamp)
            self.log_dir.mkdir()

        self.log_txt_path = pathlib.PosixPath(self.log_dir, self.timestamp + '.log')
        self.logger = Logger(self.log_txt_path)
        sys.stdout = self.logger
        sys.stderr = self.logger

        self.log_every = log_every
        self.save_models = save_models
        self.sanity = sanity
        self.batch_size = batch_size
        self.save_every = save_every

        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.writer = SummaryWriter(log_dir=self.log_dir)  # tensorboard support

        print('> Command:', ' '.join(sys.argv))
        print()

        # print current commit info
        process = subprocess.Popen(['git', 'log', '-1'], stdout=subprocess.PIPE)
        out, err = process.communicate(timeout=5)
        print(out.decode('utf-8'))

        # Set up dataloaders for the datasets
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.num_train_batches = len(self.train_loader)

    def compute_loss(self, batch):
        loss = ...
        return loss

    def train_step_end(self):
        pass

    def epoch_end(self):

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(checkpoint, PosixPath(self.log_dir, 'last_checkpoint.pt'))

        # torch.save(self.model, PosixPath(self.log_dir, 'last_model.pt'))
        # torch.save(self.optimizer, PosixPath(self.log_dir, 'last_optimizer.pt'))
        # torch.save(self.scaler, PosixPath(self.log_dir, 'last_scaler.pt'))

        last_state = dict()
        last_state['training_steps'] = self.training_steps
        last_state['epoch'] = self.epoch
        last_state['global_step'] = self.global_step

        with open(PosixPath(self.log_dir, 'last_state.json'), mode='w') as f:
            json.dump(last_state, f)

        return

    def save(self):
        if self.save_models:
            if self.epoch % self.save_every == 0:
                torch.save(self.model, pathlib.PosixPath(self.log_dir, 'epoch_{}.pt'.format(self.epoch)))

    def train(self):

        # Sanity check before training
        if self.sanity:
            self.model.eval()
            print('> perfomring a sanity check...')
            with torch.no_grad():
                self.save()  # save a copy of the untuned model
                self.epoch_end()

        # Epoch 0 is reserved for before training
        print('> start of the training loop')
        for epoch in range(self.epoch, self.num_epochs + 1):

            self.epoch = epoch

            # Training
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):

                loss = self.compute_loss(batch)

                if loss.requires_grad:

                    self.training_steps += 1
                    self.writer.add_scalar('train/steps', self.training_steps, self.global_step)

                    self.optimizer.zero_grad()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if batch_idx % self.log_every == 0:
                        self.writer.add_scalar('train/loss', loss, self.global_step)
                        self.writer.add_scalar('epoch', self.epoch, self.global_step)
                        print('train | epoch: {} | {}/{} | loss: {:.3f}'.format(
                            epoch, batch_idx, self.num_train_batches, loss
                        ))

                    self.global_step += 1

                self.train_step_end()

            # End of Epoch
            self.model.eval()
            with torch.no_grad():
                self.save()
                self.epoch_end()

            print('end of epoch {}'.format(epoch))


class T5Trainer(BaseTrainer):

    def __init__(
            self,
            *args,
            eval_every=1,
            eval_tests=None,
            thresholds=None,
            **kwargs
        ):

        self.eval_every = eval_every
        self.eval_tests = eval_tests

        self.best_metrics = {}  # map from metric (str) to the best value (float)
        self.best_path    = {}  # map from metric (str) to the path of ckpt (str)

        self.thresholds = thresholds

        return super().__init__(*args, **kwargs)

    def compute_loss(self, batch):

        outputs = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            labels=batch['labels'].to(self.device),
        )
        return outputs.loss

    def epoch_end(self):

        def _is_save_metric(metric_str):
            if 'corp_model_bleu2' in metric_str:
                return True
            if 'corp_model_ibleu2' in metric_str:
                return True
            if 'corp_model_bleu4' in metric_str:
                return True
            if 'corp_model_ibleu4' in metric_str:
                return True
            if 'dist_2' in metric_str:
                return True
            return False

        if self.epoch % self.eval_every == 0:

            with open(pathlib.PosixPath(self.log_dir, '_epoch_{}.pt.eval'.format(self.epoch)), mode='w') as f:

                results = eval_model(
                    self.eval_tests,
                    self.model,
                    tokenizer,
                    stream=f,
                    thresholds=thresholds,
                    num_dist_samples=args.num_dist_samples,
                    max_length=args.max_length,
                )

            for k, v in results.items():
                print('{}: {}'.format(k, v))
                self.writer.add_scalar(k, v, self.global_step)

            if self.save_models:

                # saving the best ckpt
                for k, v in results.items():

                    save = False

                    if _is_save_metric(k):

                        # initialization
                        if k not in self.best_metrics or k not in self.best_path:
                            save = True
                        else:
                            save = v > self.best_metrics[k]
                            # remove previous ckpt
                            if save:
                                self.best_path[k].unlink()

                        if save:
                            self.best_metrics[k] = v
                            save_name = 'best_{}_epoch_{}.pt'.format(k, self.epoch)
                            # / for organizing tensorboard, but can't use / for save path
                            save_name = save_name.replace('/', '_')
                            save_path = pathlib.PosixPath(self.log_dir, save_name)
                            torch.save(self.model, save_path)
                            self.best_path[k] = save_path

        return super().epoch_end()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training script for T5')

    parser.add_argument('--num-training-examples', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='dd', help="dailydialogue or opensubtitles")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=1)

    parser.add_argument('--train-path', type=str, default='data/dedup/train.csv')
    parser.add_argument('--eval-path', type=str, default='data/dedup/test.csv')

    parser.add_argument('--eval-max', type=int, default=None)
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--log-root-dir', type=str, default=BaseTrainer.LOG_ROOT_DIR)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-thresholds', action='store_true', help='when set, eval with all samples')
    parser.add_argument('--num-dist-samples', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=64, help='maximum utterance length (# of tokens)')
    parser.add_argument('--resume-path', default='')

    # Model parameters
    parser.add_argument('--model-str', type=str, default='t5-base')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_str)

    additional_tokens = {}
    if not tokenizer.pad_token:
        additional_tokens['pad_token'] = '<pad>'
    if not tokenizer.sep_token:
        additional_tokens['sep_token'] = '<sep>'
    tokenizer.add_special_tokens(additional_tokens)

    model = AutoModelWithLMHead.from_pretrained(args.model_str)
    model.resize_token_embeddings(len(tokenizer))

    eval_tests = build_dd_tests_from_csv(
        path=args.eval_path,
        max_num_dialogues=args.eval_max,
    )

    if args.dataset=="dd":
        dataset = DailyDialogueDataset(
            tokenizer,
            path=args.train_path,
            max_length=args.max_length,
        )
    else:
        dataset = OpenSubtitlesDataset(
            tokenizer,
            path=args.train_path,
            max_length=args.max_length,
        )

    if args.num_training_examples:
        indices = np.random.choice(len(dataset), size=args.num_training_examples, replace=False)
        train_dataset = torch.utils.data.Subset(dataset, indices=indices)
    else:
        train_dataset = dataset

    if args.no_thresholds:
        # Bascially, evaluate with everything (with overlap smaller or equal to 1)
        thresholds = [1]
    else:
        thresholds = DEFAULT_THRESHOLDS

    trainer = T5Trainer(
        model=model,
        train_dataset=train_dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        batch_size=args.batch_size,
        save_models=not args.no_save,
        log_root_dir=args.log_root_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_tests=eval_tests,
        sanity=args.sanity,
        thresholds=thresholds,
        resume_path=args.resume_path,
    )

    trainer.train()
