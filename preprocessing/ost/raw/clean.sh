# python3 clean_dialogs.py --mode=split --full-path=dedup_dialogs.txt
# python3 vector_clean.py --train-path=train.txt --valid-path=valid.txt --test-path=test.txt

# python flatten.py --input-path=../../data/proposed/dialogs/train_dialogs.txt --output-path=dirty_train.csv
# python flatten.py --input-path=../../data/proposed/dialogs/valid_dialogs.txt --output-path=dirty_valid.csv
# python flatten.py --input-path=../../data/proposed/dialogs/test_dialogs.txt --output-path=dirty_test.csv

echo "Removing self duplicates in train, valid, and test"
python ../../dd/remove_exacts.py --self --inp-path=dirty_train.csv --out-path=train.csv
python ../../dd/remove_exacts.py --self --inp-path=dirty_valid.csv --out-path=clean_valid.csv
python ../../dd/remove_exacts.py --self --inp-path=dirty_test.csv --out-path=clean_test.csv

echo "Removing cross duplicates between valid/test and train"
python ../../dd/remove_exacts.py --ref-path=train.csv --inp-path=clean_valid.csv --out-path=valid.csv
python ../../dd/remove_exacts.py --ref-path=train.csv --inp-path=clean_test.csv --out-path=test.csv
