# [An Empirical Study on the Overlapping Problem of Open-Domain Dialogue Datasets](https://arxiv.org/pdf/2201.06219.pdf)

## Cleaned Datasets

Our cleaned datasets can be downloaded at:
1. [DailyDialog](https://github.com/yq-wen/overlapping-datasets/releases/download/v0.1/cleaned_dd.zip)
2. [OpenSubtitles](https://github.com/yq-wen/overlapping-datasets/releases/download/v0.1/cleaned_ost.zip)

## Training

The training script is [train.py](https://github.com/yq-wen/overlapping-datasets/blob/main/train.py).
For example, to train the GPT-2 model on the cleaned DailyDialog dataset:
```bash
python train.py \
    --train-path=path-to-the-training-csv-file \
    --eval-path=path-to-the-validation-csv-file \
    --num-epochs=50 \
    --model-str=gpt2 \
```

LSTM and Transformer are trained using the [Fairseq](https://github.com/pytorch/fairseq) framework.

## Monitoring Performance

The training script will automatically generate a timestamped logging directory to store the checkpoints as well as log files.
The validation performance can be monitored during training through tensorboard:
```
tensorboard --logdir=path-to-the-timestamped-logging-folder
```

## Continue Training

If the performance is still increasing at the end of training, you can resume with the following command:
```bash
python train.py \
    --train-path=path-to-the-training-csv-file \
    --eval-path=path-to-the-validation-csv-file \
    --num-epochs=100 \
    --model-str=gpt2 \
    --resume-path=path-to-the-timestamped-logging-folder
```

## Evaluation

After the performance has peaked, you can evaluate the model using [eval.py](https://github.com/yq-wen/overlapping-datasets/blob/main/eval.py):
```bash
python eval.py --ckpt=path-to-the-best-validation-checkpoint --eval-path=path-to-the-test-csv-file
```
