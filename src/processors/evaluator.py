
import numpy as np
import sys

sys.path.append('../')
from utils.dataset import batch_iter, LabeledTripletDataset
from utils.math_utils import find_best_threshold


BATCHSIZE = 1000


class Evaluator(object):
    def __init__(self, metric, nbest=None, filtered=False, threshold_flg=False, all_graph=None):
        assert metric in ['mrr', 'hits', 'all', 'acc'], 'Invalid metric: {}'.format(metric)
        if metric == 'hits':
            assert nbest, 'Please indicate n-best in using hits'
        # elif metric == 'acc':
        #     self.thresholds = []
        #     self.accuracys = []
        if filtered:
            assert all_graph, 'If use filtered metric, Please indicate whole graph'
            self.all_graph = all_graph
        self.metric = metric
        self.nbest = nbest
        self.filtered = filtered
        self.threshold_flg = threshold_flg  # used for triplet classification
        if metric in ['mrr', 'hits']:
            self.threshold_flg = False
        self.batchsize = BATCHSIZE
        self.ress = []
        self.thresholds = []

    def run(self, model, dataset, test_flg=False):
        # if self.metric == 'mr':
        #     res = self.cal_mr(model, dataset)
        if self.metric == 'mrr':
            res = self.cal_mrr(model, dataset)
        elif self.metric == 'hits':
            res = self.cal_hits(model, dataset, self.nbest)
        elif self.metric == 'acc':
            res = self.cal_accuracy(model, dataset, test_flg)
        else:
            raise ValueError
        self.ress.append(res)
        return res

    def run_all_matric(self, model, dataset):
        """
        calculating MRR, Hits@1,3,10 (raw and filter)
        """
        n_sample = len(dataset)
        sum_rr_raw = 0.
        sum_rr_flt = 0.
        n_corr_h1_raw = 0
        n_corr_h1_flt = 0
        n_corr_h3_raw = 0
        n_corr_h3_flt = 0
        n_corr_h10_raw = 0
        n_corr_h10_flt = 0
        start_id = 0
        for samples in batch_iter(dataset, self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))

            # TODO: partitioned calculation
            # search objects
            raw_scores = model.cal_scores(subs, rels)
            raw_ranks = self.cal_rank(raw_scores, objs)
            sum_rr_raw += sum(float(1/rank) for rank in raw_ranks)
            n_corr_h1_raw += sum(1 for rank in raw_ranks if rank <=1)
            n_corr_h3_raw += sum(1 for rank in raw_ranks if rank <=3)
            n_corr_h10_raw += sum(1 for rank in raw_ranks if rank <=10)
            # filter
            if self.filtered:
                flt_scores = self.cal_filtered_score_fast(subs, rels, objs, ids, raw_scores)
                flt_ranks = self.cal_rank(flt_scores, objs)
                sum_rr_flt += sum(float(1/rank) for rank in flt_ranks)
                n_corr_h1_flt += sum(1 for rank in flt_ranks if rank <=1)
                n_corr_h3_flt += sum(1 for rank in flt_ranks if rank <=3)
                n_corr_h10_flt += sum(1 for rank in flt_ranks if rank <=10)

            # seach subjects
            raw_scores_inv = model.cal_scores_inv(rels, objs)
            raw_ranks_inv = self.cal_rank(raw_scores_inv, subs)
            sum_rr_raw += sum(float(1/rank) for rank in raw_ranks_inv)
            n_corr_h1_raw += sum(1 for rank in raw_ranks_inv if rank <=1)
            n_corr_h3_raw += sum(1 for rank in raw_ranks_inv if rank <=3)
            n_corr_h10_raw += sum(1 for rank in raw_ranks_inv if rank <=10)
            # filter
            if self.filtered:
                flt_scores_inv = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, raw_scores_inv)
                flt_ranks_inv = self.cal_rank(flt_scores_inv, subs)
                sum_rr_flt += sum(float(1/rank) for rank in flt_ranks_inv)
                n_corr_h1_flt += sum(1 for rank in flt_ranks_inv if rank <=1)
                n_corr_h3_flt += sum(1 for rank in flt_ranks_inv if rank <=3)
                n_corr_h10_flt += sum(1 for rank in flt_ranks_inv if rank <=10)

            start_id += len(samples)

        return {'MRR': sum_rr_raw/n_sample/2,
                'Hits@1': n_corr_h1_raw/n_sample/2,
                'Hits@3': n_corr_h3_raw/n_sample/2,
                'Hits@10': n_corr_h10_raw/n_sample/2,
                'MRR(filter)': sum_rr_flt/n_sample/2,
                'Hits@1(filter)': n_corr_h1_flt/n_sample/2,
                'Hits@3(filter)': n_corr_h3_flt/n_sample/2,
                'Hits@10(filter)': n_corr_h10_flt/n_sample/2}

    def cal_mrr(self, model, dataset):
        n_sample = len(dataset)
        sum_rr = 0.
        start_id = 0
        for samples in batch_iter(dataset, self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))
            scores = model.cal_scores(subs, rels)
            if self.filtered:
                scores = self.cal_filtered_score_fast(subs, rels, objs, ids, scores)
            ranks1 = self.cal_rank(scores, objs)

            scores = model.cal_scores_inv(rels, objs)
            if self.filtered:
                scores = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, scores)
            ranks2 = self.cal_rank(scores, subs)
            sum_rr += sum(float(1/rank) for rank in ranks1 + ranks2)
            start_id += len(samples)
        return float(sum_rr/n_sample/2)

    def cal_hits(self, model, dataset, nbest):
        n_sample = len(dataset)
        n_corr = 0
        start_id = 0
        for samples in batch_iter(dataset, self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))
            scores = model.cal_scores(subs, rels)
            if self.filtered:
                scores = self.cal_filtered_score_fast(subs, rels, objs, ids, scores)
            res = np.flip(np.argsort(scores), 1)[:, :nbest]
            n_corr += sum(1 for i in range(len(objs)) if objs[i] in res[i])

            scores = model.cal_scores_inv(rels, objs)
            if self.filtered:
                scores = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, scores)
            res = np.flip(np.argsort(scores), 1)
            n_corr += sum(1 for i in range(len(subs)) if subs[i] in res[i])
            start_id += len(samples)
        return float(n_corr/n_sample/2)

    def cal_accuracy(self, model, dataset, test_flg):
        assert isinstance(dataset, LabeledTripletDataset)
        if self.threshold_flg:
            if test_flg:
                res = self._cal_accuracy_with_threshold
            else:
                res = self._cal_accuracy_with_threshold(model, dataset)
        else:
            res = self._cal_sigmoid_accuracy(model, dataset)
        return res

    def _cal_sigmoid_accuracy(self, model, dataset):
        n_sample = len(dataset)
        n_corr = 0
        for samples in batch_iter(dataset, self.batchsize, rand_flg=False):
            raw_scores = model.cal_triplet_scores(samples[:, :3])
            pred_labels = [1 if s > 0 else -1 for s in raw_scores]
            n_corr += sum(1 for i in range(len(samples)) if pred_labels[i] == samples[i][3])
        return float(n_corr/n_sample)

    def _cal_accuracy_with_threshold(self, model, dataset):
        acc_each = []
        threshold_each = []
        n_each = []
        n_corr = 0
        for idx in range(model.n_relation):
            dataset_each = LabeledTripletDataset(dataset.samples[np.where(dataset.samples[:, 1] == idx)])
            labels = []
            scores = []
            for xys in batch_iter(dataset_each, self.batchsize, rand_flg=False):
                samples, _labels = xys[:, :-1], xys[:, 3]
                scores += model.cal_triplet_scores(samples).tolist()
                labels += _labels.tolist()
            threshold, accuracy = find_best_threshold(scores, labels)
            threshold_each.append(threshold)
            acc_each.append(accuracy)
            n_each.append(len(labels))
            n_corr += int(len(labels) * accuracy)
        self.thresholds.append(threshold_each)
        print(threshold_each)
        print(acc_each)
        return float(n_corr/sum(n_each))

    def cal_filtered_score_fast(self, subs, rels, objs, ids, raw_scores, metric='sim'):
        assert metric in ['sim', 'dist']
        new_scores = []
        for s, r, o, i, score in zip(subs, rels, objs, ids, raw_scores):
            true_os = self.id2obj_list[i]
            true_os_rm_o = np.delete(true_os, np.where(true_os==o))
            if metric=='sim':
                score[true_os_rm_o] = -np.inf
            else:
                score[true_os_rm_o] = np.inf
            new_scores.append(score)
        return new_scores

    def cal_filtered_score_inv_fast(self, subs, rels, objs, ids, raw_scores, metric='sim'):
        assert metric in ['sim', 'dist']
        new_scores = []
        for s, r, o, i, score in zip(subs, rels, objs, ids, raw_scores):
            true_ss = self.id2sub_list[i]
            true_ss_rm_s = np.delete(true_ss, np.where(true_ss==s))
            if metric=='sim':
                score[true_ss_rm_s] = -np.inf
            else:
                score[true_ss_rm_s] = np.inf
            new_scores.append(score)
        return new_scores

    def cal_rank(self, score_mat, ents, filtered=False):
        # return [1 + np.sum(score > score[e]) for score, e in zip(score_mat, ents)]
        return [np.sum(score >= score[e]) for score, e in zip(score_mat, ents)]


    def get_best_info(self):
        if self.metric == 'mrr' or self.metric == 'hits' or self.metric == 'acc':  # higher value is better
            best_val = max(self.ress)
        elif self.metric == 'mr':
            best_val = min(self.ress)
        else:
            raise ValueError('Invalid')
        best_epoch = self.ress.index(best_val) + 1
        return best_epoch, best_val

    def prepare_valid(self, dataset):
        self.id2sub_list = []
        self.id2obj_list = []
        self.sr2o = {}
        self.ro2s = {}
        for i in range(len(dataset)):
            s, r, o = dataset.samples[i]
            os = self.all_graph.search_obj_id(s, r)
            ss = self.all_graph.search_sub_id(r, o)
            self.id2obj_list.append(os)
            self.id2sub_list.append(ss)
            self.sr2o[(s, r)] = os
            self.ro2s[(r, o)] = ss

    # === CAUTION: this method save the LAST thresholds===
    def save_threshold(self, thresh_path):
        with open(thresh_path, 'w') as fw:
            print(self.thresholds[-1], file=fw)

    def load_threshold(self, thresh_path):
        raise NotImplementedError
