
import numpy as np
import pickle
import sys

sys.path.append('../')
from utils.math_utils import *


ENT_PARAM = ['e_re', 'e_im']

class Optimizer(object):
    def update(self):
        if hasattr(self, 'l2_coeff'):
            self._l2_addhook()
        elif hasattr(self, 'ent_l2_coeff'):
            self._ent_l2_addhook
        if hasattr(self, 'gc_norm'):
            self._gradclip_addhook()
        self._update()

    def _l2_addhook(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                for i, idx in enumerate(param.grad_idxs):
                    param.part_grads[i] += 2 * self.l2_coeff * param.data[idx]
            else:
                raise NotImplementedError

    def _ent_l2_addhook(self):
        for p_name in self.params.keys():
            if p_name in ENT_PARAM:
                param = self.params[p_name]
                if type(param) == LookupParameter:
                    for i, idx in enumerate(param.grad_idxs):
                        param.part_grads[i] += 2 * self.l2_coeff * param.data[idx]
                else:
                    raise NotImplementedError

    def _gradclip_addhook(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                for i, idx in enumerate(param.grad_idxs):
                    norm = np.linalg.norm(param.part_grads[i])
                    if norm > self.gc_norm:
                        param.part_grads[i] *= self.gc_norm / norm
            else:
                raise NotImplementedError

    def regist_params(self, params):
        self.params = params
        self._prepare()

    def set_l2_reg(self, coeff):
        self.l2_coeff = coeff

    def sel_ent_l2_reg(self, coeff):
        self.ent_l2_coeff = coeff

    def set_gradclip(self, norm):
        self.gc_norm = norm

    def _prepare(self):
        pass

    def save_opt(self, opt_path):
        with open(opt_path, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load_opt(cls, opt_path):
        with open(opt_path, 'rb') as f:
            opt = pickle.load(f)
        return opt


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def _update(self):
        for param in self.params.values():
            if type(param) == LookupParameter:
                idxs = param.grad_idxs
                # print(param.part_grads)
                if len(idxs) != 0:
                    param.data[idxs] -= self.lr * np.array(param.part_grads)
            else:
                param.data -= self.lr * param.grad

class Adagrad(Optimizer):
    def __init__(self, lr, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.grad_history = {}

    def _update(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if type(param) == LookupParameter:
                idxs = param.grad_idxs
                if len(idxs) != 0:
                    self.grad_history[p_name][idxs] += np.power(param.part_grads, 2)
                    param.data[idxs] -= self.lr * np.array(param.part_grads) / (np.sqrt(self.grad_history[p_name][idxs]) + self.eps)
            else:
                self.grad_history[p_name] += np.power(param.grad, 2)
                param.data -= self.lr * param.grad / (np.sqrt(self.grad_history[p_name]) + self.eps)

    def _init_grad_history(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            self.grad_history[p_name] = np.zeros_like(param.data)

    def _prepare(self):
        self._init_grad_history()


class AdagradRDA(Optimizer):
    def __init__(self, lr, l1_reg, eps=1e-8):
        """
        args:
          - lr (float): learning rate
          - l1_reg (float): coefficient in L1 reguralizer
        """
        self.lr = lr
        self.l1_reg = l1_reg
        self.prev_ave_grads = {}  # for memorizing previous averaging gradients
        self.update_counts = {}
        self.eps = eps
        self.grad_history = {}

    def set_reg_param(self, param_names):
        self.param_names = param_names

    def _init_prev_ave_grads(self):
        for p_name in self.param_names:
            param = self.params[p_name]
            self.prev_ave_grads[p_name] = np.zeros_like(param.data)

    def _init_update_counts(self):
        for p_name in self.param_names:
            param = self.params[p_name]
            if type(param) == LookupParameter:
                self.update_counts[p_name] = np.zeros((len(param.data, )), dtype=np.int32)
            else:
                raise NotImplementedError
                self.update_counts[p_name] = 0

    def _update(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            if p_name in self.param_names:  # considering regularization term
                if type(param) == LookupParameter:
                    idxs = param.grad_idxs
                    if len(idxs) != 0:
                        self.update_counts[p_name][idxs] += 1
                        self.grad_history[p_name][idxs] += np.power(param.part_grads, 2)
                        _counts = np.expand_dims(self.update_counts[p_name][idxs], axis=1)
                        _prev_ave_grads = self.prev_ave_grads[p_name][idxs]
                        _cur_ave_grads = (1/_counts)*param.part_grads + ((_counts-1)/_counts)*_prev_ave_grads
                        _param = self.lr*_counts*(- _cur_ave_grads + self.l1_reg*np.sign(_cur_ave_grads)) / (np.sqrt(self.grad_history[p_name][idxs]) + self.eps)
                        assert len(idxs) == len(_param)
                        _param[np.abs(_cur_ave_grads) <= self.l1_reg] = 0.

                        param.data[idxs] = _param
                        self.prev_ave_grads[p_name][idxs] = _cur_ave_grads
                else:
                    raise NotImplementedError

            else:  # usual update
                if type(param) == LookupParameter:
                    idxs = param.grad_idxs
                    if len(idxs) != 0:
                        self.grad_history[p_name][idxs] += np.power(param.part_grads, 2)
                        param.data[idxs] -= self.lr * np.array(param.part_grads) / (np.sqrt(self.grad_history[p_name][idxs]) + self.eps)
                else:
                    raise NotImplementedError
                    param.data -= self.lr * param.grad

    def _init_grad_history(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            self.grad_history[p_name] = np.zeros_like(param.data)

    def _prepare(self):
        self._init_prev_ave_grads()
        self._init_update_counts()
        self._init_grad_history()


class AdagradRDAmul(AdagradRDA):
    def __init__(self, lr, l1_reg, eps=1e-8):
        """
        args:
          - lr (float): learning rate
          - l1_reg (float): coefficient in L1 reguralizer
        """
        super(AdagradRDAmul, self).__init__(lr=lr, l1_reg=l1_reg)
        self.eps = eps
        self.grad_history = {}

    def set_reg_param(self, param_names):
        assert param_names == ['r_im', 'r_re'] or param_names == ['r_re', 'r_im'], 'This optimizer is only available to reguralize r_im and r_re'
        self.param_names = param_names

    def _update(self):
        assert self.params['r_re'].grad_idxs == self.params['r_im'].grad_idxs
        r_idxs = self.params['r_re'].grad_idxs
        for p_name in self.params.keys():
            param = self.params[p_name]
            if p_name == 'r_re' or p_name == 'r_im':
                assert type(param) == LookupParameter
                if len(r_idxs) != 0:
                    if p_name == 'r_re':
                        _l1_reg = self.l1_reg * np.abs(self.params['r_im'].data[r_idxs])
                    else:  # r_im
                        _l1_reg = self.l1_reg * np.abs(self.params['r_re'].data[r_idxs])
                    self.update_counts[p_name][r_idxs] += 1
                    self.grad_history[p_name][r_idxs] += np.power(param.part_grads, 2)
                    up_cnt = np.expand_dims(self.update_counts[p_name][r_idxs], axis=1)
                    prev_gs = self.prev_ave_grads[p_name][r_idxs]
                    cur_gs = (1 / up_cnt) * param.part_grads + ((up_cnt-1) / up_cnt) * prev_gs
                    new_param = self.lr * up_cnt * (-cur_gs + _l1_reg * np.sign(cur_gs)) / (np.sqrt(self.grad_history[p_name][r_idxs]) + self.eps)
                    new_param[np.abs(cur_gs) <= _l1_reg] = 0.
                    if p_name == 'r_re':
                        new_r_re = new_param
                    else:
                        new_r_im = new_param
                    self.prev_ave_grads[p_name][r_idxs] = cur_gs

            else:  # usual update
                if type(param) == LookupParameter:
                    idxs = param.grad_idxs
                    if len(idxs) != 0:
                        self.grad_history[p_name][idxs] += np.power(param.part_grads, 2)
                        param.data[idxs] -= self.lr * np.array(param.part_grads) / (np.sqrt(self.grad_history[p_name][idxs]) + self.eps)
                else:
                    raise NotImplementedError
                    param.data -= self.lr * param.grad

        if len(r_idxs) != 0:
            self.params['r_re'].data[r_idxs] = new_r_re
            self.params['r_im'].data[r_idxs] = new_r_im

    def _init_grad_history(self):
        for p_name in self.params.keys():
            param = self.params[p_name]
            self.grad_history[p_name] = np.zeros_like(param.data)

    def _prepare(self):
        self._init_prev_ave_grads()
        self._init_update_counts()
        self._init_grad_history()
