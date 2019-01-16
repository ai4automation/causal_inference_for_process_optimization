import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import os
from tqdm import tqdm


class GenerateData(object):
    def __init__(self, args):
        self.samples_n = args['n']
        self.samples_m = args['m']

    def sample_from_gaussian_mixture(self):
        k_i = np.random.random_integers(1, 5)
        r_i = np.random.uniform(0, 5)
        s_i = np.random.uniform(0, 5)

        mean_list = np.random.normal(0, r_i, k_i)
        variance_list = np.abs(np.random.normal(0, s_i, k_i) + 3)
        mixture_weights = np.abs(np.random.normal(0, 1, k_i) + 2)

        mixture_weights = mixture_weights / sum(mixture_weights)
        assert sum(mixture_weights) - 1 < 1e-6

        x_i = []
        for i in range(self.samples_m):
            mixture_number = np.random.choice(k_i, p=mixture_weights)
            x_i.append(np.random.normal(mean_list[mixture_number], variance_list[mixture_number]))

        x_i = (np.array(x_i) - np.mean(x_i)) / np.std(x_i)
        return x_i

    def generate_one_bag(self, x_i):
        k_i = len(x_i)
        d_i = np.random.random_integers(4, 5)
        x_std = np.std(x_i)

        # assume knots are equally spaced
        knots_x = np.linspace(np.min(x_i) - x_std, np.max(x_i) + x_std, num=d_i)
        knots_y = np.random.normal(0, 1, d_i)

        f_i = interp1d(knots_x, knots_y, kind='cubic')

        assert len(x_i) == self.samples_m

        # find y_i and normalize
        y_i = f_i(x_i)
        y_i = (y_i - np.mean(y_i)) / np.std(y_i)

        # generate noise
        v_i = np.random.uniform(0, 5)
        noise_e_ij = np.random.normal(0, v_i, k_i)
        knots_v_ij = np.random.uniform(0, 5, d_i)

        noise_spline = interp1d(knots_x, knots_v_ij, kind='cubic')
        x_samples_for_noise_multiplier = np.random.uniform(np.min(x_i) - x_std, np.max(x_i) + x_std, k_i)
        noise_multipliers = noise_spline(x_samples_for_noise_multiplier)

        noise = np.multiply(noise_e_ij, noise_multipliers)

        y_i = y_i + noise
        y_i = (y_i - np.mean(y_i)) / np.std(y_i)

        return y_i

    def generate_and_save(self, save_path):
        if not os.path.exists(save_path):
            print('Creating directory to save data -', save_path)
            os.makedirs(save_path)

        print('Generating samples for %d bags with %d samples each' % (self.samples_n, self.samples_m))

        for i in tqdm(range(self.samples_n)):
            x_i = self.sample_from_gaussian_mixture()
            y_i = self.generate_one_bag(x_i)

            df = pd.DataFrame({'X': x_i, 'Y': y_i})
            csv_filename = os.path.join(save_path, str(i+1)+'.csv')
            df.to_csv(csv_filename, index=False)


if __name__ == '__main__':
    TOTAL_SAMPLES = 10000
    SAMPLES_PER_BAG = 1000

    SAVE_PATH = '/Users/tanmayee/Documents/dev/bpm-causal/data'

    data_generator = GenerateData({'n': TOTAL_SAMPLES, 'm': SAMPLES_PER_BAG})
    data_generator.generate_and_save(SAVE_PATH)
