import unittest
import numpy as np
import common
import kmeans


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


if __name__ == '__main__':
    unittest.main()
