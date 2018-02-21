
from collections import defaultdict
import math
import numpy as np
import operator


class LookupParameter(object):
    def __init__(self, name, shape, init_method='xavier'):
        self.name = name
        if init_method == 'xavier':
            self.data = xavier_init(shape)
        else:
            raise NotImplementedError
        self.grad_idxs = []
        self.part_grads = []
        self.dim = shape[1]
        self.idx2grad = defaultdict(lambda: np.zeros(self.dim))
        self.grad_idxs = self.idx2grad.keys()
        self.part_grads = self.idx2grad.values()

    def add_grad(self, idx, grad):
        self.idx2grad[idx] += grad

    def add_all_grads(self, idxs, grads):
        [self.add_grad(i, g) for i, g in zip(idxs, grads)]

    def clear(self):
        self.grad_idxs = []
        self.part_grads = []
        self.idx2grad = defaultdict(lambda: np.zeros(self.dim))

    def finalize(self):
        self.grad_idxs = list(self.idx2grad.keys())
        self.part_grads = list(self.idx2grad.values())


class LookupParameter2(object):
    def __init__(self, name, shape, init_method='xavier'):
        self.name = name
        if init_method == 'xavier':
            self.data = xavier_init(shape)
        else:
            raise NotImplementedError
        self.grad_idxs = set()
        self.part_grads = []

    def add_grad(self, idx, part_grad):
        self.grad_idxs.add(idx)
        self.grad[idx] += part_grad

    def add_all_grads(self, idxs, part_grads):
        # [self.add_grad(i, g) for i, g in zip(idxs, part_grads)]
        np.add.at(self.grad, (idxs.tolist(), slice(None)), part_grads)
        # self.grad[list(self.grad_idxs)]
        # print('a')

    def clear(self):
        self.grad_idxs = set()
        self.grad = np.zeros_like(self.data)


def xavier_init(size):
    assert len(size) < 4
    if len(size) == 3:
        assert size[1] == size[2]
    dim = size[1]
    bound = math.sqrt(6) / math.sqrt(2*dim)
    return np.random.uniform(-bound, bound, size=size)


def max_margin(pos_scores, neg_scores):
    return np.maximum(0, 1 - (pos_scores - neg_scores))


def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5


def softplus(x):
    return np.maximum(0,x)+np.log(1+np.exp(-np.abs(-x)))


def find_best_threshold(scores, labels, debug=False):
    # find best threshold in O(nlogn)
    # does not handle scores of infinity or -infinity

    total = len(scores)
    total_pos = len([l for l in labels if l==1])

    def accuracy(p, n):
        correct_n = n
        correct_p = total_pos - p
        return float(correct_n + correct_p) / total

    # predict True iff score > thresh
    pos = 0  # no. pos <= thresh
    neg = 0  # no. neg <= thresh

    thresh_accs = [(float('-inf'), accuracy(pos, neg))]
    for idx in np.argsort(scores):
        thresh = scores[idx]
        label = labels[idx]
        if label==1:
            pos += 1
        else:
            neg += 1
        thresh_accs.append((thresh, accuracy(pos, neg)))
    return max(thresh_accs, key=operator.itemgetter(1))


if __name__ == '__main__':
    l = LookupParameter('test', (10000, 200))
    batchsize = 2000
    for e in range(10):
        idxs = np.random.randint(low=0, high=10000, size=(batchsize))
        gs = np.random.random(size=(batchsize, 200))
        for i, g in zip(idxs, gs):
            l.add_grad(i, g)
