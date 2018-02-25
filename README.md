# symmetry-learning-kgc

Learning the degree of symmetry via sparse modeling for Knowledge Graph Embedding

## Setup

```
git clone https://github.com/mana-ysh/symmetry-learning-kgc.git
cd symmetry-learning-kgc/
pip install -r requirements.txt
```

## How to Run

* Running ```preprocess.sh``` and ```run_{fb | wn}_{trouillon | std | mul}.sh``` in ```symmetry-learning-kgc/scripts```
* Then, generated the log directory in ```symmetry-learning-kgc/src/main```

## Arguments

For training

```
⟩⟩⟩ python train.py -h
usage: Link prediction models [-h] [--mode MODE] [--task TASK] [--ent ENT]
                              [--rel REL] [--train TRAIN] [--valid VALID]
                              [--method METHOD] [--restart RESTART]
                              [--epoch EPOCH] [--batch BATCH] [--lr LR]
                              [--dim DIM] [--margin MARGIN]
                              [--negative NEGATIVE] [--opt OPT]
                              [--l2_reg L2_REG] [--gradclip GRADCLIP]
                              [--save_step SAVE_STEP] [--metric METRIC]
                              [--nbest NBEST] [--filtered]
                              [--graphall GRAPHALL] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           training mode ["pairwise", "single"]
  --task TASK           link prediction task ["kbc", "tc"]
  --ent ENT             entity list
  --rel REL             relation list
  --train TRAIN         training data
  --valid VALID         validation data
  --method METHOD       method ["complex"]
  --restart RESTART     retraining model path
  --epoch EPOCH         number of epochs
  --batch BATCH         batch size
  --lr LR               learning rate
  --dim DIM             dimension of embeddings
  --margin MARGIN       margin in max-margin loss for pairwise training
  --negative NEGATIVE   number of negative samples for pairwise training
  --opt OPT             optimizer ["sgd", "adagrad"]
  --l2_reg L2_REG       L2 regularization
  --gradclip GRADCLIP   gradient clipping
  --save_step SAVE_STEP
  --metric METRIC       evaluation metrics ["mrr", "hits", "acc"]
  --nbest NBEST         n-best for hits metric
  --filtered            use filtered metric
  --graphall GRAPHALL   all graph file for filtered evaluation
  --log LOG             output log dir
```

```
⟩⟩⟩ python train_sparse.py -h
usage: Link prediction models [-h] [--mode MODE] [--task TASK] [--ent ENT]
                              [--rel REL] [--train TRAIN] [--valid VALID]
                              [--method METHOD] [--restart RESTART]
                              [--epoch EPOCH] [--batch BATCH] [--lr LR]
                              [--dim DIM] [--margin MARGIN]
                              [--negative NEGATIVE] [--reg REG]
                              [--l1_ratio L1_RATIO] [--opt OPT]
                              [--gradclip GRADCLIP] [--add_re]
                              [--save_step SAVE_STEP] [--metric METRIC]
                              [--nbest NBEST] [--filtered]
                              [--graphall GRAPHALL] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           training mode ["pairwise", "single"]
  --task TASK           link prediction task ["kbc", "tc"]
  --ent ENT             entity list
  --rel REL             relation list
  --train TRAIN         training data
  --valid VALID         validation data
  --method METHOD       method ["complex"]
  --restart RESTART     retraining model path
  --epoch EPOCH         number of epochs
  --batch BATCH         batch size
  --lr LR               learning rate
  --dim DIM             dimension of embeddings
  --margin MARGIN       margin in max-margin loss for pairwise training
  --negative NEGATIVE   number of negative samples for pairwise training
  --reg REG             strength of L1/L2 regularization
  --l1_ratio L1_RATIO   ratio of L1
  --opt OPT             optimizer ["adarda", "adardamul"]
  --gradclip GRADCLIP   gradient clipping
  --add_re
  --save_step SAVE_STEP
  --metric METRIC       evaluation metrics ["mrr", "hits", "acc"]
  --nbest NBEST         n-best for hits metric
  --filtered            use filtered metric
  --graphall GRAPHALL   all graph file for filtered evaluation
  --log LOG             output log dir
```

For Testing

```
⟩⟩⟩ python test.py -h
usage: Link prediction models [-h] [--task TASK] [--ent ENT] [--rel REL]
                              [--data DATA] [--filtered] [--graphall GRAPHALL]
                              [--method METHOD] [--model MODEL]
                              [--metric METRIC]

optional arguments:
  -h, --help           show this help message and exit
  --task TASK          link prediction task ["kbc", "tc"]
  --ent ENT            entity list
  --rel REL            relation list
  --data DATA          test data
  --filtered           use filtered metric
  --graphall GRAPHALL  all graph file for filtered evaluation
  --method METHOD      method ["transe", "complex", "analogy"]
  --model MODEL        trained model path
  --metric METRIC      evaluation metric
```


## Dependencies

See ```requirements.txt```




## Reference

- Manabe, H.; Hayashi, K.; and Shimbo, M. 2018. Data-dependent Learning of Symmetric/Antisymmetric Relations for Knowledge Base Completion. In AAAI18.
