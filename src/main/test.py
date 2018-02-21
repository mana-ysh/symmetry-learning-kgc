
import argparse
from datetime import datetime
import logging
import os
import sys
import tensorflow as tf

sys.path.append('../')
from processors.evaluator import Evaluator
from utils.dataset import LabeledTripletDataset, TripletDataset, Vocab
from utils.graph import *


def test(args):
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)

    # preparing data
    if args.task == 'kbc':
        test_dat = TripletDataset.load(args.data, ent_vocab, rel_vocab)
    elif args.task == 'tc':
        test_dat = LabeledTripletDataset.load(args.data, ent_vocab, rel_vocab)
    else:
        raise ValueError('Invalid task: {}'.format(args.task))

    print('loading model...')
    if args.method == 'transe':
        from models.transe import TransE as Model
    elif args.method == 'complex':
        from models.complex import ComplEx as Model
    elif args.method == 'analogy':
        from models.analogy import ANALOGY as Model
    else:
        raise NotImplementedError

    if args.filtered:
        print('loading whole graph...')
        from utils.graph import TensorTypeGraph
        graphall = TensorTypeGraph.load_from_raw(args.graphall, ent_vocab, rel_vocab)
        # graphall = TensorTypeGraph.load(args.graphall)
    else:
        graphall = None

    model = Model.load_model(args.model)

    if args.metric == 'all':
        evaluator = Evaluator('all', None, args.filtered, False, graphall)
        if args.filtered:
            evaluator.prepare_valid(test_dat)

        all_res = evaluator.run_all_matric(model, test_dat)
        for metric in sorted(all_res.keys()):
            print('{:20s}: {}'.format(metric, all_res[metric]))
    else:
        evaluator = Evaluator(args.metric, None, False, True, None)
        res = evaluator.run(model, test_dat)
        print('{:20s}: {}'.format(args.metric, res))


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.add_argument('--task', default='kbc', type=str, help='link prediction task ["kbc", "tc"]')

    # dataset
    p.add_argument('--ent', type=str, help='entity list')
    p.add_argument('--rel', type=str, help='relation list')
    p.add_argument('--data', type=str, help='test data')
    p.add_argument('--filtered', action='store_true', help='use filtered metric')
    p.add_argument('--graphall', type=str, help='all graph file for filtered evaluation')

    # model
    p.add_argument('--method', default=None, type=str,
                   help='method ["transe", "complex", "analogy"]')
    p.add_argument('--model', type=str, help='trained model path')

    # evaluation metric
    p.add_argument('--metric', default='all', type=str, help='evaluation metric')

    args = p.parse_args()
    test(args)
