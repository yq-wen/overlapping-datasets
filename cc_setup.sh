module load python/3.6 && ENVDIR=/tmp/$RANDOM && virtualenv --no-download $ENVDIR && source $ENVDIR/bin/activate && pip install --no-index --upgrade pip

pip install --no-index nltk
pip install --no-index torch
pip install --no-index transformers

python download.py --model="t5-base"
python download.py --model="t5-small"

python -m nltk.downloader all

deactivate
rm -rf $ENVDIR
