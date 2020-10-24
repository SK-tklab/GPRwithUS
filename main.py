import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve
import seaborn as sns

sns.set_style('darkgrid')


class GP:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, noise_var: float = 1., lscale: float = 1.,
                 k_var: float = 1., prior_mean: float = 0, standardize=True):
        self.__lscale = lscale
        self.__k_var = k_var
        self.__noise_var = noise_var
        self.__x_train = x_train
        self.__y_train = y_train
        self.__n_train = y_train.shape[0]
        if standardize:
            self.__prior_mean = np.mean(y_train)
        else:
            self.__prior_mean = prior_mean
        self.__standardize = standardize

        self.__K_inv = None
        self.__set_k_inv()

    @property
    def n_train(self):
        return self.__n_train

    @property
    def lscale(self):
        return self.__lscale

    @property
    def k_var(self):
        return self.__k_var

    @property
    def noise_var(self):
        return self.__noise_var

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    def gauss_kernel(self, x1, x2):
        assert x1.ndim == 2
        assert x2.ndim == 2
        r = np.linalg.norm(x1[:, None] - x2, axis=2)
        return self.k_var * np.exp(-0.5 * np.square(r) / np.square(self.lscale))

    def __set_k_inv(self):
        K = self.gauss_kernel(self.x_train, self.x_train)
        K += self.noise_var * np.eye(self.n_train)
        self.__K_inv = cho_solve((cholesky(K, True), True), np.eye(self.n_train))

    def predict(self, x):
        assert x.ndim == 2
        kx = self.gauss_kernel(self.x_train, x)  # (n,m)
        kK = kx.T @ self.__K_inv
        mean = kK @ (self.y_train - self.__prior_mean)
        var = self.gauss_kernel(x, x) + self.noise_var * np.eye(x.shape[0]) - kK @ kx
        return mean.flatten() + self.__prior_mean, np.diag(var)

    def add_observation(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.__x_train = np.vstack([self.__x_train, x])
        self.__y_train = np.vstack([self.__y_train, y])
        self.__n_train = self.__y_train.shape[0]
        self.__set_k_inv()
        if self.__standardize:
            self.__prior_mean = np.mean(self.y_train)
        return


def f(x):
    return x + x * np.sin(5 * x)


def main():
    x_train = np.array([0, 0.5, 1.2]).reshape(-1, 1)
    y_train = f(x_train)

    gp = GP(x_train, y_train, lscale=0.4, noise_var=1e-4)
    x_test_plot = np.linspace(-1, 2, 201)
    x_test = x_test_plot.reshape(-1, 1)
    test_f = f(x_test_plot)

    max_itr = 8

    for i in range(max_itr):
        m, v = gp.predict(x_test)
        std = np.sqrt(v)
        next_x = x_test[np.argmax(v)].reshape(1, 1)
        next_y = f(next_x)

        fig, ax = plt.subplots()
        ax.plot(x_test_plot, test_f, ls='--', c='black', lw=2, label=r'$f(x)$')
        ax.plot(x_test_plot, m, ls='-', c='tab:blue', lw=2, label='predict mean')
        ax.fill_between(x_test_plot, m + 1.96 * std, m - 1.96 * std, fc='tab:blue', alpha=0.2, label='95% ci')
        ax.scatter(gp.x_train.flatten(), gp.y_train.flatten(), ec='black', c='gold', s=50, marker='s', label='observed',        zorder=5)
        ax.axvline(next_x.flatten(), ls='--', lw=2, c='crimson', label='next', zorder=5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$f(x)$')
        ax.set_title(f'iteration:{i + 1}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'./image/iterarion{i + 1}.png')
        gp.add_observation(next_x, next_y)
        plt.close('all')
    return


if __name__ == '__main__':
    main()
