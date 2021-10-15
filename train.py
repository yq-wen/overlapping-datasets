import argparse
import torch
import datetime
import pathlib
import sys
import subprocess

from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dailydialogue import DailyDialogueDataset
from dateutil import tz
from eval import eval_model
from util import build_dd_test_dict


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
        eval=False,
    ):

        torch.manual_seed(0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.to(self.device)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.log_every = log_every
        self.save_models = save_models
        self.sanity = sanity
        self.eval = eval
        self.batch_size = batch_size

        # Set up for logging
        if not log_root_dir:
            log_root_dir = BaseTrainer.LOG_ROOT_DIR
        self.log_root_dir = pathlib.PosixPath(log_root_dir)
        if not self.log_root_dir.exists():
            self.log_root_dir.mkdir()

        tzone = tz.gettz('America/Edmonton')
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
        self.log_dir = pathlib.PosixPath(self.log_root_dir, self.timestamp)
        self.log_dir.mkdir()

        self.log_txt_path = pathlib.PosixPath(self.log_dir, self.timestamp + '.log')
        self.logger = Logger(self.log_txt_path)
        sys.stdout = self.logger
        sys.stderr = self.logger

        self.writer = SummaryWriter(log_dir=self.log_dir)  # tensorboard support

        self.training_steps = 0
        self.epoch = 0
        self.global_step = 0

        # Set up for optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print('> Command:', ' '.join(sys.argv))
        print()

        # print current commit info
        process = subprocess.Popen(['git', 'log', '-1'], stdout=subprocess.PIPE)
        out, err = process.communicate(timeout=5)
        print(out.decode('utf-8'))

        # Set up dataloaders for the datasets
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        if self.eval:
            self.eval_loader = DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.num_train_batches = len(self.train_loader)

    def compute_loss(self, batch):
        loss = ...
        return loss

    def train_step_end(self):
        pass

    def epoch_end(self):
        # evaluation
        pass

    def save(self):
        if self.save_models:
            torch.save(self.model, pathlib.PosixPath(self.log_dir, 'epoch_{}.pt'.format(self.epoch)))

    def train(self):

        scaler = torch.cuda.amp.GradScaler()

        # Sanity check before training
        if self.sanity:
            self.model.eval()
            print('> perfomring a sanity check...')
            with torch.no_grad():
                self.save()  # save a copy of the untuned model
                self.epoch_end()

        # Epoch 0 is reserved for before training
        print('> start of the training loop')
        for epoch in range(1, self.num_epochs + 1):

            self.epoch = epoch

            # Training
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):

                loss = self.compute_loss(batch)

                if loss.requires_grad:

                    self.training_steps += 1
                    self.writer.add_scalar('train/steps', self.training_steps, self.global_step)

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    self.writer.add_scalar('epoch', self.epoch, self.global_step)

                    self.global_step += 1

                    if batch_idx % self.log_every == 0:
                        print('train | epoch: {} | {}/{} | loss: {:.3f}'.format(
                            epoch, batch_idx, self.num_train_batches, loss
                        ))

                self.train_step_end()

            # Evaluation steps
            if self.eval:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch_idx, batch in enumerate(self.eval_loader):
                        loss = self.compute_loss(batch)
                        val_losses.append(loss.item())
                    val_loss = statistics.mean(val_losses)
                    self.writer.add_scalar('val/loss', val_loss, self.global_step)

            # End of Epoch
            with torch.no_grad():
                self.save()
                self.epoch_end()

            if self.eval:
                print('end of epoch {} | val loss: {:.3f}'.format(epoch, val_loss))
            else:
                print('end of epoch {}'.format(epoch))

class T5Trainer(BaseTrainer):

    def compute_loss(self, batch):

        outputs = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            labels=batch['labels'].to(self.device),
        )
        return outputs.loss

    def epoch_end(self):

        with open(pathlib.PosixPath(self.log_dir, '_epoch_{}.pt.eval'.format(self.epoch)), mode='w') as f:
            results = eval_model(TEST_DICT, self.model, tokenizer, stream=f)

        for k, v in results.items():
            print('{}: {}'.format(k, v))
            self.writer.add_scalar(k, v, self.global_step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training script for T5')

    args = parser.parse_args()

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-5)

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelWithLMHead.from_pretrained("t5-base")

    TEST_DICT = build_dd_test_dict(
        path='data/clean_dailydialog/validation/dialogues_validation_clean.txt',
        max_num_dialogues=100
    )

    train_dataset = DailyDialogueDataset(tokenizer)

    trainer = T5Trainer(
        model=model,
        train_dataset=train_dataset,
        num_epochs=1000,
        learning_rate=5e-5,
        log_every=100,
        batch_size=64,
        save_models=True,
        log_root_dir=None,
    )

    trainer.train()
