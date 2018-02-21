
import copy
import numpy as np
import os
import sys
import time

sys.path.append('../')
from utils.dataset import *
from processors.optimizer import *

np.random.seed(46)

REGPARAMS = ['r_im']


class Trainer(object):
    def __init__(self):
        pass

    def _setup(self):
        pass

    def fit(self):
        pass


class PairwiseTrainer(object):
    def __init__(self, model, opt, **kwargs):
        self.model = model
        self.n_entity = model.n_entity
        self.n_relation = model.n_relation
        self.opt = opt
        self.n_epoch = kwargs.pop('epoch')
        self.batchsize = kwargs.pop('batchsize')
        self.logger = kwargs.pop('logger')
        self.log_dir = kwargs.pop('model_dir')
        self.evaluator = kwargs.pop('evaluator')
        self.valid_dat = kwargs.pop('valid_dat')
        self.n_negative = kwargs.pop('n_negative')
        self.pretrain_model_path = kwargs.pop('restart')
        self.save_step = kwargs.pop('save_step')
        self.add_re_flg = kwargs.pop('add_re', False)
        self.sample_strategy = kwargs.pop('sample_strategy', 'uniform')
        if self.add_re_flg:
            REGPARAMS.append('r_re')
        self.model_path = os.path.join(self.log_dir, self.model.__class__.__name__)

        """
        TODO: make other negative sampling methods available. and assuming only objects are corrupted
        """
        if self.sample_strategy == 'uniform':
            self.neg_sampler = UniformIntSampler(0, self.n_entity)
        elif self.sample_strategy == 'unigram':
            self.neg_sampler = None
        elif self.sample_strategy == 'unirel':
            self.neg_sampler = UniformWholeIntSampler(self.n_entity, self.n_relation)
        else:
            raise NotImplementedError

    def fit(self, samples):
        self.logger.info('setup trainer...')
        if self.sample_strategy == 'unigram':
            self.neg_sampler = UnigramIntSampler.build(samples, self.n_entity)

        if type(self.opt) == AdagradRDA:
            if self.add_re_flg:
                self.opt.set_reg_param(REGPARAMS + ['r_re'])
            else:
                self.opt.set_reg_param(REGPARAMS)
        elif type(self.opt) == AdagradRDAmul:
            self.opt.set_reg_param(['r_im', 'r_re'])
        self.opt.regist_params(self.model.params)

        self.best_model = None
        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for batch in batch_iter(samples, self.batchsize):
                # setup for mini-batch training
                _batchsize = len(batch)
                n_samples = _batchsize * self.n_negative
                pos_triples = np.tile(batch, (self.n_negative, 1))
                neg_triplets = self.neg_sampler.generate(batch, self.n_negative)
                pos_triples = np.tile(batch, (self.n_negative, 1))
                loss = self.model.compute_gradients(pos_triples, neg_triplets)
                self.opt.update()
                sum_loss += loss

            if self.valid_dat:  # run validation
                valid_start = time.time()
                res = self.evaluator.run(self.model, self.valid_dat)
                self.logger.info('evaluation metric in {} epoch: {}'.format(epoch+1, res))
                self.logger.info('evaluation time in {} epoch: {}'.format(epoch+1, time.time()-valid_start))
                if self.evaluator.threshold_flg:
                    self.evaluator.save_threshold(os.path.join(self.log_dir, 'threshold.epoch{}'.format(epoch+1)))

                cur_best_epoch, cur_best_val = self.evaluator.get_best_info()
                self.logger.info('< Current Best metric: {} ({} epoch) >'.format(cur_best_val, cur_best_epoch))
                if cur_best_epoch == epoch+1:
                    self.best_model = copy.deepcopy(self.model)

            if (epoch+1) % self.save_step == 0:
                self.model.save_model(self.model_path+'.epoch{}'.format(epoch+1))

            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

        self.best_model.save_model(self.model_path+'.best')
        if self.valid_dat:
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))


class SingleTrainer(PairwiseTrainer):
    def __init__(self, model, opt, **kwargs):
        super(SingleTrainer, self).__init__(model, opt, **kwargs)

    def fit(self, samples):
        if self.sample_strategy == 'unigram':
            self.neg_sampler = UnigramIntSampler.build(samples, self.n_entity)

        if type(samples) == TripletDataset:
            self._fit_negative_sample(samples)
        elif type(samples) == LabeledTripletDataset:
            self._fit_labeled(samples)
        else:
            raise

    def _fit_negative_sample(self, samples):
        self.logger.info('setup trainer...')

        if type(self.opt) == AdagradRDA:
            if self.add_re_flg:
                self.opt.set_reg_param(REGPARAMS + ['r_re'])
            else:
                self.opt.set_reg_param(REGPARAMS)
        elif type(self.opt) == AdagradRDAmul:
            self.opt.set_reg_param(['r_im', 'r_re'])
        self.opt.regist_params(self.model.params)

        self.best_model = None
        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for batch in batch_iter(samples, self.batchsize):
                assert batch.shape[1] == 3
                # setup for mini-batch training
                _batchsize = len(batch)
                n_samples = _batchsize * self.n_negative
                neg_triplets = self.neg_sampler.generate(batch, self.n_negative)
                ys = np.concatenate((np.ones(_batchsize), -np.ones(n_samples)))
                loss = self.model.compute_gradients(np.r_[batch, neg_triplets], ys)
                self.opt.update()
                sum_loss += loss

            if self.valid_dat:  # run validation
                valid_start = time.time()
                res = self.evaluator.run(self.model, self.valid_dat)
                self.logger.info('evaluation metric in {} epoch: {}'.format(epoch+1, res))
                self.logger.info('evaluation time in {} epoch: {}'.format(epoch+1, time.time()-valid_start))
                if self.evaluator.threshold_flg:
                    self.evaluator.save_threshold(os.path.join(self.log_dir, 'threshold.epoch{}'.format(epoch+1)))

                cur_best_epoch, cur_best_val = self.evaluator.get_best_info()
                self.logger.info('< Current Best metric: {} ({} epoch) >'.format(cur_best_val, cur_best_epoch))
                if cur_best_epoch == epoch+1:
                    self.best_model = copy.deepcopy(self.model)

            if (epoch+1) % self.save_step == 0:
                if self.best_model:
                    self.best_model.save_model(self.model_path+'.epoch{}'.format(cur_best_epoch))
                else:
                    self.model.save_model(self.model_path+'.epoch{}'.format(epoch+1))

            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

        if self.best_model:
            self.best_model.save_model(self.model_path+'.best')
        else:
            self.model.save_model(self.model_path+'.epoch{}'.format(epoch+1))
        if self.valid_dat:
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))

    def _fit_labeled(self, samples):
        self.logger.info('setup trainer...')

        if type(self.opt) == AdagradRDA:
            if self.add_re_flg:
                self.opt.set_reg_param(REGPARAMS + ['r_re'])
            else:
                self.opt.set_reg_param(REGPARAMS)
        elif type(self.opt) == AdagradRDAmul:
            self.opt.set_reg_param(['r_im', 'r_re'])
        self.opt.regist_params(self.model.params)

        self.best_model = None
        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for batch in batch_iter(samples, self.batchsize):
                # setup for mini-batch training
                _batchsize = len(batch)
                loss = self.model.compute_gradients(batch[:, :3], batch[:, -1])
                self.opt.update()
                sum_loss += loss

            if self.valid_dat:  # run validation
                valid_start = time.time()
                res = self.evaluator.run(self.model, self.valid_dat)
                self.logger.info('evaluation metric in {} epoch: {}'.format(epoch+1, res))
                self.logger.info('evaluation time in {} epoch: {}'.format(epoch+1, time.time()-valid_start))
                if self.evaluator.threshold_flg:
                    self.evaluator.save_threshold(os.path.join(self.log_dir, 'threshold.epoch{}'.format(epoch+1)))

                cur_best_epoch, cur_best_val = self.evaluator.get_best_info()
                self.logger.info('< Current Best metric: {} ({} epoch) >'.format(cur_best_val, cur_best_epoch))
                if cur_best_epoch == epoch+1:
                    self.best_model = copy.deepcopy(self.model)

            if (epoch+1) % self.save_step == 0:
                self.best_model.save_model(self.model_path+'.epoch{}'.format(cur_best_epoch))

            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

        self.best_model.save_model(self.model_path+'.best')
        if self.valid_dat:
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))


class UniformIntSampler(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def generate(self, pos_triplets, n_negative):
        """
        "n_negative" means number of negative samples
        """
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * n_negative
        neg_ents = self.sample(sample_size)
        neg_triplets = np.tile(pos_triplets, (n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets

    def sample(self, size):
        return np.random.randint(self.lower, self.upper, size=size)


class UnigramIntSampler(object):
    def __init__(self, lower, upper, uni_count):
        self.lower = lower
        self.upper = upper
        # assert sum(dist) == 1.0, "sum(dist) is {}".format(sum(dist))  # numerical error
        # self.dist = dist
        self.uni_count = uni_count
        self.sortid2id = np.argsort(uni_count)
        self.sorted_uni_count = copy.deepcopy(uni_count)
        self.sorted_uni_count.sort()
        self._build_cum_table()

    def _build_cum_table(self, power=0.75, domain=2**31 - 1):
        self.cum_table = np.zeros(len(self.sorted_uni_count), dtype=np.int32)
        train_words_pow = np.power(self.sorted_uni_count, power).sum()
        cumulative = 0.0
        for word_index in range(len(self.sorted_uni_count)):
            cumulative += self.sorted_uni_count[word_index]**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def generate(self, pos_triplets, n_negative):
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * n_negative
        neg_ents = self.sample(sample_size)
        neg_triplets = np.tile(pos_triplets, (n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets

    def sample(self, size):
        assert type(size) == int
        return np.array([self.one_sample() for _ in range(size)])

    def one_sample(self):
        _id = self.cum_table.searchsorted(np.random.randint(self.cum_table[-1]))
        return self.sortid2id[_id]

    @classmethod
    def build(cls, samples, upper):
        cnt = np.zeros(upper)
        for sample in batch_iter(samples, 1):
            s, r, o = sample[0][:3]
            cnt[s] += 1
            cnt[o] += 1
        return UnigramIntSampler(0, upper, cnt)


class UniformWholeIntSampler(object):
    def __init__(self, n_ent, n_rel):
        self.n_ent = n_ent
        self.n_rel = n_rel

    def generate(self, pos_triplets, n_negative):
        """
        "n_negative" means number of negative samples
        CAUTION : this relation sampling method is NOT UNIFORM
        """
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * n_negative
        neg_rel_ents = self.sample(sample_size)
        neg_triplets = np.tile(pos_triplets, (n_negative, 1))
        head_rel_tail = np.random.randint(0, 3, sample_size)
        rel_idxs = np.where(head_rel_tail==1)
        # TODO: fix to sample uniformly
        neg_rel_ents[rel_idxs] = neg_rel_ents[rel_idxs] % self.n_rel
        neg_triplets[np.arange(sample_size), head_rel_tail] = neg_rel_ents
        return neg_triplets

    def sample(self, size):
        return np.random.randint(0, self.n_ent, size=size)


if __name__ == '__main__':
    from collections import defaultdict
    from utils.dataset import TripletDataset
    d = TripletDataset([[0, 0, 1],[0, 0, 2], [1, 0, 1]])
    sampler = UnigramIntSampler.build(d, 3)

    dic = defaultdict(lambda: 0)
    for _ in range(5):
        sample = sampler.sample(10)
        print(sample)
        for i in sample:
            dic[i] += 1
    print(dic)
