
import argparse
from datetime import datetime
import logging
import os
import sys

sys.path.append('../')
from processors.trainer import PairwiseTrainer, SingleTrainer
from processors.evaluator import Evaluator
from processors.optimizer import SGD, Adagrad
from utils.dataset import TripletDataset, LabeledTripletDataset, Vocab, generate_synthetic
from utils.graph import TensorTypeGraph


MODEL_CONFIG_FILE = 'model.config'
DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))


def train(args):
    # setting for logging
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.log, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in sorted(vars(args).items()):
        logger.info('{:>10} -----> {}'.format(arg, val))

    if args.synthetic:
        train, valid, test = generate_synthetic(args.n_ent, args.n_rel, args.pos_ratio, args.sym_ratio, split=True)
        train_dat = TripletDataset(train)
        valid_dat = TripletDataset(valid)
        n_entity = args.n_ent
        n_relation = args.n_rel

        # generate negative samples
        g = TensorTypeGraph(train+valid+test, args.n_ent, args.n_rel)
        neg_samples = []
        for r in range(n_relation):
            neg_samples += [[s, r, o] for s, o in zip(*np.where(g.rel2mat[r].todense()==0))]

    else:
        ent_vocab = Vocab.load(args.ent)
        rel_vocab = Vocab.load(args.rel)
        n_entity, n_relation = len(ent_vocab), len(rel_vocab)

        # knowledge base completion
        if args.task == 'kbc':
            train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
            valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None
        # triplet classification
        elif args.task == 'tc':
            assert args.metric == 'acc'
            train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
            valid_dat = LabeledTripletDataset.load(args.valid, ent_vocab, rel_vocab) if args.valid else None
        else:
            raise ValueError('Invalid task: {}'.format(args.task))

    if args.opt == 'sgd':
        opt = SGD(args.lr)
    elif args.opt == 'adagrad':
        opt = Adagrad(args.lr)
    else:
        raise NotImplementedError

    if args.l2_reg > 0:
        opt.set_l2_reg(args.l2_reg)
    if args.gradclip > 0:
        opt.set_gradclip(args.gradclip)

    logger.info('building model...')
    if args.method == 'complex':
        from models.complex import ComplEx
        model = ComplEx(n_entity=n_entity,
                        n_relation=n_relation,
                        margin=args.margin,
                        dim=args.dim,
                        mode=args.mode)
    else:
        raise NotImplementedError

    if args.filtered:
        print('loading whole graph...')
        from utils.graph import TensorTypeGraph
        graphall = TensorTypeGraph.load_from_raw(args.graphall, ent_vocab, rel_vocab)
    else:
        graphall = None
    evaluator = Evaluator(args.metric, args.nbest, args.filtered, True, graphall) if args.valid or args.synthetic else None
    if args.filtered and args.valid:
        evaluator.prepare_valid(valid_dat)
    if args.mode == 'pairwise':
        trainer = PairwiseTrainer(model=model, opt=opt, save_step=args.save_step,
                                  batchsize=args.batch, logger=logger,
                                  evaluator=evaluator, valid_dat=valid_dat,
                                  n_negative=args.negative, epoch=args.epoch,
                                  model_dir=args.log, restart=args.restart,
                                  sample_strategy=args.sample)
    elif args.mode == 'single':
        trainer = SingleTrainer(model=model, opt=opt, save_step=args.save_step,
                                batchsize=args.batch, logger=logger,
                                evaluator=evaluator, valid_dat=valid_dat,
                                n_negative=args.negative, epoch=args.epoch,
                                model_dir=args.log, restart=args.restart,
                                sample_strategy=args.sample)
    else:
        raise NotImplementedError

    trainer.fit(samples=train_dat)


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.add_argument('--mode', default='pairwise', type=str, help='training mode ["pairwise", "single"]')
    p.add_argument('--task', default='kbc', type=str, help='link prediction task ["kbc", "tc"]')

    # real world dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--train', type=str, help='training data')
    p.add_argument('--valid', type=str, help='validation data')

    # synthetic data
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--labeled', action='store_true')
    p.add_argument('--n_rel', default=10, type=int, help='number of relation')
    p.add_argument('--n_ent', default=100, type=int, help='number of entity')
    p.add_argument('--pos_ratio', default=0.1, type=float, help='ratio of obserbed positive triplet')
    p.add_argument('--sym_ratio', default=0.5, type=float, help='ratio of symmetry reation')

    # model
    p.add_argument('--method', default='complex', type=str,
                   help='method ["complex", "gencomplex", "transe", "distmult", "analogy"]')
    p.add_argument('--restart', default=None, type=str, help='retraining model path')
    p.add_argument('--epoch', default=100, type=int, help='number of epochs')
    p.add_argument('--batch', default=128, type=int, help='batch size')
    p.add_argument('--lr', default=0.001, type=float, help='learning rate')
    p.add_argument('--dim', default=100, type=int, help='dimension of embeddings')
    p.add_argument('--margin', default=1., type=float, help='margin in max-margin loss for pairwise training')
    p.add_argument('--negative', default=10, type=int, help='number of negative samples for pairwise training')
    p.add_argument('--opt', default='sgd', type=str, help='optimizer ["sgd", "adagrad"]')
    p.add_argument('--l2_reg', default=0., type=float, help='L2 regularization')
    p.add_argument('--gradclip', default=-1, type=float, help='gradient clipping')
    p.add_argument('--save_step', default=100, type=int)
    p.add_argument('--sample', default='uniform', type=str)


    # model-specific config
    p.add_argument('--comp', default='conv', type=str, help='compositional function in HolE ["conv", "corr"]')
    p.add_argument('--cp_ratio', default=0.5, type=float, help="ratio of complex's dimention in ANALOGY")
    p.add_argument('--shift', default=1.0, type=float, help='shifted-value for Shift ComplEx')

    # evaluation
    p.add_argument('--metric', default='mrr', type=str, help='evaluation metrics ["mrr", "hits", "acc"]')
    p.add_argument('--nbest', default=None, type=int, help='n-best for hits metric')
    p.add_argument('--filtered', action='store_true', help='use filtered metric')
    p.add_argument('--graphall', type=str, help='all graph file for filtered evaluation')

    # others
    p.add_argument('--log', default=DEFAULT_LOG_DIR, type=str, help='output log dir')

    args = p.parse_args()
    train(args)
