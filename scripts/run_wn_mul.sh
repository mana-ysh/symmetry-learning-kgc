
cd ../src/main

python train_sparse.py \
  --train ../../dat/wordnet-mlj12/wordnet-mlj12-train.txt \
  --ent ../../dat/wordnet-mlj12/train.entlist \
  --rel ../../dat/wordnet-mlj12/train.rellist \
  --method complex \
  --log ./log-wn-mul \
  --l1_ratio 0.3 \
  --negative 5 \
  --dim 200 \
  --lr 0.1 \
  --reg 0.001 \
  --epoch 500 \
  --valid ../../dat/wordnet-mlj12/wordnet-mlj12-valid.txt \
  --metric mrr \
  --opt adardamul \
  --mode single \
  --graphall ../../dat/wordnet-mlj12/wordnet-mlj12-test.txt \
  --filtered \
  --gradclip 5


python test.py \
  --method complex \
  --ent ../../dat/wordnet-mlj12/train.entlist \
  --rel ../../dat/wordnet-mlj12/train.rellist \
  --data ../../dat/wordnet-mlj12/wordnet-mlj12-test.txt \
  --model ./log-wn-mul/ComplEx.best \
  --graphall ../../dat/wordnet-mlj12/whole.txt \
  --filtered > ./log-wn-mul/test_results.txt
