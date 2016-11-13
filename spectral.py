#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import warnings
import itertools

from sklearn.cluster import SpectralClustering, spectral_clustering
# from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np


class ESpectralClustering(SpectralClustering):
    def __init__(self,  n_clusters=8, eigen_solver=None, random_state=None,
            n_init=10, gamma=1., affinity='cosine', n_neighbors=10,
            eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
            kernel_params=None, n_jobs=1, weight=np.array([]), merge='product'):
        super(ESpectralClustering, self).__init__(n_clusters=n_clusters,
                eigen_solver=eigen_solver,
                random_state=random_state, n_init=n_init, gamma=gamma,
                affinity=affinity,
                n_neighbors=n_neighbors, eigen_tol=eigen_tol,
                assign_labels=assign_labels, degree=degree,
                coef0=coef0, kernel_params=kernel_params, n_jobs=n_jobs)
        self.weight = weight
        self.merge = merge

    def calc_weight(self, w1, w2):
        # self.merge = 'product' or 'min'
        if self.merge == 'product':
            weight = np.array([float(w1[i]) * float(w2[i])
                for i in xrange(len(w1))])
            return weight / sum(weight)
        else:
            raise

    def fit(self, X, y=None):
        '''
            X is matrix((n_separate * n) * len(feature[0]))
            w is matrix(n * n_separate)

        '''
        # X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
        #                 dtype=np.float64)
        if X.shape[0] == X.shape[1] and self.affinity != 'precomputed':
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "set ``affinity=precomputed``.")
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.affinity):
            params['gamma'] = self.gamma
            params['degree'] = self.degree
            params['coef0'] = self.coef0
        # X = np.random.randint(1, 7, (10, 2))
        # affity = 'cosine'
        if len(self.weight) == 0:
            self.affinity_matrix_ = pairwise_kernels(X,
                    metric=self.affinity, filter_params=True, **params)
        else:
            n_data = len(self.weight)
            n_separate = len(self.weight[0])
            print 'n_data: ', n_data
            print 'se', n_separate
            # zerosでよかった...
            D = np.array([[[0.0000000 for ___ in range(len(X[0]))]
                for __ in range(n_data)] for _ in range(n_separate)])
            for i in xrange(n_data * n_separate):
                D[i % n_separate, i / n_separate] = X[i]
            mat_W = np.zeros([len(self.weight), len(self.weight),
                len(self.weight[0])])
            for i, j in itertools.product(range(len(self.weight)),
                    range(len(self.weight))):
                print 'i:', self.weight[i]
                print 'j:', self.weight[j]
                mat_W[i, j] = self.calc_weight(self.weight[i], self.weight[j])
            print 'W: ', mat_W
            kernel_D = np.array([pairwise_kernels(mat, metric=self.affinity)
                for mat in D])
            sum_kernel_D = np.zeros((len(self.weight), len(self.weight)))
            for i, j in itertools.product(range(len(self.weight)),
                    range(len(self.weight))):
                temp = 0
                for k in range(len(kernel_D)):
                    temp += kernel_D[k, i, j] * mat_W[i, j, k]
                sum_kernel_D[i, j] = temp
            self.affinity_matrix_ = sum_kernel_D

        print 'result: ', self.affinity_matrix_

        random_state = check_random_state(self.random_state)
        print 'affinity mat: ', self.affinity_matrix_
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self


if __name__ == '__main__':
    np.random.seed(10)
    data = np.random.randint(1, 7, (30, 2))
    print('-' * 10 + 'sklearn' + '-' * 10)
    I = SpectralClustering()
    assign_labels = I.fit_predict(data)
    print 'affinity', I.affinity
    print(assign_labels)

    np.random.seed(10)
    data = np.random.randint(1, 7, (30, 2))
    print('-' * 10 + 'sklearn' + '-' * 10)
    I2 = SpectralClustering(affinity='cosine')
    assign_labels = I2.fit_predict(data)
    print 'affinity', I2.affinity
    print(assign_labels)
    print('-' * 10 + 'E' + '-' * 10)
    np.random.seed(10)
    E = ESpectralClustering()
    e_assign_labels = E.fit_predict(data)
    print 'affinity', E.affinity
    print(e_assign_labels)
    print('-' * 10 + 'E with weight' + '-' * 10)
    np.random.seed(10)
    weight = abs(np.random.randn(15, 2))
    print data
    E = ESpectralClustering(n_clusters=2, weight=weight)
    e_assign_labels = E.fit_predict(data)
    print 'affinity', E.affinity
    print(e_assign_labels)
