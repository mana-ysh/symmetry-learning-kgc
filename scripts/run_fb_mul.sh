
cd ../src/main

python train_sparse.py \
  --train ../../dat/FB15k/freebase_mtr100_mte100-train.txt \
  --ent ../../dat/FB15k/train.entlist \
  --rel ../../dat/FB15k/train.rellist \
  --method complex \
  --log ./log-fb-mul \
  --l1_ratio 0.7 \
  --negative 10 \
  --dim 200 \
  --lr 0.05 \
  --reg 0.001 \
  --epoch 500 \
  --valid ../../dat/FB15k/freebase_mtr100_mte100-valid.txt \
  --metric mrr \
  --opt adardamul \
  --mode single \
  --graphall ../../dat/FB15k/whole.txt \
  --filtered \
  --gradclip 5


python test.py \
  --method complex \
  --ent ../../dat/FB15k/train.entlist \
  --rel ../../dat/FB15k/train.rellist \
  --data ../../dat/FB15k/freebase_mtr100_mte100-test.txt \
  --model ./log-fb-mul/ComplEx.best \
  --graphall ../../dat/FB15k/whole.txt \
  --filtered > ./log-fb-mul/test_results.txt
