
import dill
import numpy as np
import pickle
import yaml


class BaseModel(object):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def max_margin(self, pos_scores, neg_scores):
        return np.maximum(0, 1 - (pos_scores - neg_scores))

    def cal_rank(self, sample):
        raise NotImplementedError

    def _pairwisegrads(self, pos_samples, neg_samples):
        raise NotImplementedError

    def _singlegrads(self, xs, ys):
        raise NotImplementedError

    def _composite(self, sub, rel):
        raise NotImplementedError

    def _cal_similarity(self, query, obj):
        raise NotImplementedError

    def pick_ent(self, ents):
        raise NotImplementedError

    def pick_rel(self, rels):
        raise NotImplementedError

    def cal_scores(self, query_idxs):
        raise NotImplementedError

    def zerograds(self):
        for param in self.params.values():
            param.clear()

    def reset_memory(self):
        self.fw_mem = {}

    def prepare(self):
        self.zerograds()
        self.reset_memory()

    def save_model(self, model_path):
        with open(model_path, 'wb') as fw:
            dill.dump(self, fw)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            try:
                model = pickle.load(f)
            except:
                model = dill.load(f)
        return model
