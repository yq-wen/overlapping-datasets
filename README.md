# few-shot-dialogue


Train the Dailydialogue dataset
```sh
$ python3 train.py --dataset dd --train-path data/dedup/train.csv --eval-path data/dedup/test.csv --model-str t5-base
```

Train the Opensubtitles dataset
```sh
$ python3 train.py --dataset ost --train-path data/data_ost/df_ost_train_v2.csv --eval-path data/data_ost/test_ost_compare.csv --model-str t5-base
```
