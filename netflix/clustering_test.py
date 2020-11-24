import unittest
import numpy as np
import common
import kmeans
import naive_em
import em


class ClusteringTest(unittest.TestCase):
    def test_k_means(self):
        X = np.loadtxt("toy_data.txt")

        K_list = [1, 2, 3, 4]
        seeds = [0, 1, 2, 3, 4]
        print('\n')

        for seed in seeds:
            print(f'Seed = {seed}\n')
            for K in K_list:
                mixture, post = common.init(X, K, seed)
                mixture, post, cost = kmeans.run(X, mixture, post)
                print(f'Cost for {K} clusters = {cost}')
            print('\n')
        self.assertEqual(True, True)

    def test_em_naive(self):
        X = np.loadtxt("toy_data.txt")

        K_list = [1, 2, 3, 4]
        seeds = [0, 1, 2, 3, 4]
        print('\n')

        for seed in seeds:
            print(f'Seed = {seed}\n')
            for K in K_list:
                mixture, post = common.init(X, K, seed)
                mixture, post, log_likelihood = naive_em.run(X, mixture, post)
                bic = common.bic(X, mixture, log_likelihood)
                print(f'Log likelihood for {K} clusters = {log_likelihood}, BIC = {bic}')
            print('\n')

        self.assertEqual(True, True)

    def test_kmeans_em_naive(self):
        X = np.loadtxt("toy_data.txt")

        K_list = [1, 2, 3, 4]
        seeds = [0, 1, 2, 3, 4]
        print('\n')

        for seed in seeds:
            print(f'Seed = {seed}\n')
            for K in K_list:
                mixture, post = common.init(X, K, seed)
                mixture, post, cost = kmeans.run(X, mixture, post)
                print(f'Cost for {K} clusters = {cost}')
                common.plot(X, mixture, post, f'KMeans : {K} clusters, Seed = {seed}')

                mixture, post = common.init(X, K, seed)
                mixture, post, cost = naive_em.run(X, mixture, post)
                print(f'Cost for {K} clusters = {cost}')
                common.plot(X, mixture, post, f'EM : {K} clusters, Seed = {seed}')
            print('\n')

        self.assertEqual(True, True)

    def test_em(self):
        X = np.loadtxt("netflix_incomplete.txt")

        K_list = [5, 12]
        seeds = [0, 1, 2, 3, 4]
        print('\n')

        for seed in seeds:
            print(f'Seed = {seed}\n')
            for K in K_list:
                mixture, post = common.init(X, K, seed)
                mixture, post, log_likelihood = em.run(X, mixture, post)
                filled_matrix = em.fill_matrix(X, mixture)
                print(f'Log likelihood for {K} clusters = {log_likelihood}')
            print('\n')

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
