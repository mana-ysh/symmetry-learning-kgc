
cd ../src/main

python train.py \
  --train ../../dat/FB15k/freebase_mtr100_mte100-train.txt \
  --ent ../../dat/FB15k/train.entlist \
  --rel ../../dat/FB15k/train.rellist \
  --method complex \
  --log ./log-fb-trouillon \
  --negative 10 \
  --dim 200 \
  --lr 0.05 \
  --l2_reg 0.0001 \
  --epoch 500 \
  --valid ../../dat/FB15k/freebase_mtr100_mte100-valid.txt \
  --metric mrr \
  --opt adagrad \
  --mode single \
  --graphall ../../dat/FB15k/whole.txt \
  --filtered \
  --gradclip 5

python test.py \
  --method complex \
  --ent ../../dat/FB15k/train.entlist \
  --rel ../../dat/FB15k/train.rellist \
  --data ../../dat/FB15k/freebase_mtr100_mte100-test.txt \
  --model ./log-fb-trouillon/ComplEx.best \
  --graphall ../../dat/FB15k/whole.txt \
  --filtered > ./log-fb-trouillon/test_results.txt
