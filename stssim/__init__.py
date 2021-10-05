import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def summary_plot(time_series, return_type="absolute"):
    """Plot a visual summary of a time series.

    Include the time series, returns, and return distribution.

    Args:
        time_series (array): The time series as a list, tuple, or numpy array.
        return_type (string): Either "absolute" or "percent".

    Returns:
        None
    """

    if return_type == 'percent':
        returns = np.insert(np.diff(time_series), 0, 0) / time_series * 100
    else:
        returns = np.insert(np.diff(time_series), 0, 0)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_series)
    ax1.set_title('Series')
    ax1.tick_params(right=True, labelright=True)
    ax1.annotate(
        f"n={len(time_series)}",
        xy=(.03, .95),
        xycoords='axes fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )

    ax2 = fig.add_subplot(gs[1, :-1])
    ax2.plot(returns)
    ax2.set_title(f"Returns ({return_type})")
    
    mean = np.mean(returns)
    std = np.std(returns)
    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    ax3 = fig.add_subplot(gs[1, -1])
    ax3.hist(returns, bins=int(len(returns) / 20), orientation="horizontal")
    ax3.set_title('Distribution')
    ax3.tick_params(left=False, right=True, labelleft=False, labelright=True)
    annotation = "\n".join([
        "$\mu_1$=" + str(round(mean, 4)),
        "$\sqrt{\mu_2}$=" + str(round(std, 4)),
        "$\mu_3$=" + str(round(skew, 4)),
        "$\mu_4$=" + str(round(kurtosis, 4)),
    ])
    ax3.annotate(
        annotation,
        xy=(.95, .95),
        xycoords='axes fraction',
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )
    
    return fig

class Series:
    def __init__(self, series=None):
        self.series = series
        if self.series:
            self.returns = np.insert(np.diff(self.series), 0, 0)

    def plot(self, filepath, return_type="absolute"):
        plt.clf()
        f = summary_plot(self.series, return_type)
        f.savefig(filepath)
        plt.clf()

    def bin(self, cardinality=2, split=None):
        if split is None:
            split = self.mean

    def from_dependencies(self, dependencies, generator, generates="returns", no_negative=True, x0=0.0):
        """Create a new series dependent on other series.

        Creates a new series with a dependency on other series defined by the generator function.
        The generator function is responsible for handling boundary conditions for its dependency series.
    
        Args:
            xxx (int): Desc.
    
        Returns:
            None: The return value.
        """

        if not all([i.series.shape == dependencies[0].series.shape for i in dependencies]):
            raise ValueError(f"Dependencies have mismatched shapes: {', '.join([str(i.series.shape) for i in dependencies])}")

        if generates == "series":
            self.series = np.array([generator(dependencies, i) for i, x in enumerate(dependencies[0].series)])
            self.returns = np.insert(np.diff(self.series), 0, 0)
        else:
            self.returns = np.array([generator(dependencies, i) for i, x in enumerate(dependencies[0].returns)])
            if no_negative:
                self.series = [x0]
                for i, x in enumerate(self.returns[1:]):
                    self.series.append(max(x + self.series[i], 0.0))
                self.series = np.array(self.series)
                self.returns = np.insert(np.diff(self.series), 0, 0)
            else:
                self.series = x0 + self.returns.cumsum()



class MarkovSeries(Series):
    def __init__(self):
        super.__init__()


class StochasticSeries(Series):

    def __init__(self, n, distribution=stats.norm(loc=0, scale=1), x0=0.0, no_negative=True):
        """Create a new stochastic time series

        Create a new time series from a Markov process generated from a given random probability distribution.
        Note that any child time series that are created from this parent that inject dependencies on previous
        values of itself or of its parent will not be Markov processes since future values will now depend on 
        previous states of the system.

        Args:
            n (int): Number of points in series.
            distribution (scipy dist): A distribution from the scipy.stats class list.
            x0 (float): Value of the first point.
            no_negative (bool): If True, only non-negative values will be permitted.
                Negative values generated will be rounded to zero.
        """


        self.n = n
        self.distribution = distribution
        self.returns = np.insert(self.distribution.rvs(self.n - 1), 0, 0.0)
        if no_negative:
            self.series = [x0]
            for i, x in enumerate(self.returns[1:]):
                self.series.append(max(x + self.series[i], 0.0))
            self.series = np.array(self.series)
            self.returns = np.insert(np.diff(self.series), 0, 0)
        else:
            self.series = x0 + self.returns.cumsum()

    

    


if __name__ == "__main__":

    # # Set random seed for reproducibility.
    # np.random.seed(seed=23342379)
    # N = 10000

    # # Some example distributions.
    # t = stats.nct(nc=.02, df=2.7)
    # norm = stats.norm(loc=0.1, scale=20)
    # uniform = stats.uniform(loc=-50, scale=100)



    # # Create a non-negative time series of N steps
    # s1 = StochasticSeries(N, distribution=t, x0=1, no_negative=True)

    # # Plot the time series.
    # s1.plot(pathlib.Path.home() / 'time_series.png', return_type='absolute')


    # # Create a new series linearly dependent on the first.
    # s2 = Series()
    # lag = 4
    # s2.from_dependencies([s1], generator=lambda dependencies, i: 2 * (0 if i < lag else dependencies[0].returns[i - lag]) + 2*(np.random.rand() - .5), generates="returns", x0=1)
    # s2.plot(pathlib.Path.home() / 'time_series2.png', return_type='absolute')

    # # Examine Pearson correlation coefficient.
    # r = stats.pearsonr(s1.returns, s2.returns)
    # print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}")


    # # Create a new series non-linearly dependent on the first.
    # lag = 7
    # s3 = Series()
    # s3.from_dependencies([s1], generator=lambda dependencies, i: 0.0 if i < lag else np.sin(3 * dependencies[0].returns[i - lag]), generates="returns", x0=100)
    # s3.plot(pathlib.Path.home() / 'time_series3.png', return_type='absolute')

    # # Examine Pearson correlation coefficient.
    # r = stats.pearsonr(s1.returns, s3.returns)
    # print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}")


    # # Create a new series with no dependencies.
    # s4 = StochasticSeries(N, distribution=norm, x0=100, no_negative=True)
    # s4.plot(pathlib.Path.home() / 'time_series4.png', return_type='absolute')

    # # Examine Pearson correlation coefficient.
    # r = stats.pearsonr(s1.returns, s4.returns)
    # print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}")



    # # TE Estimation
    # # Import classes
    # from idtxl.multivariate_te import MultivariateTE
    # from idtxl.bivariate_te import BivariateTE
    # from idtxl.data import Data
    # from idtxl.visualise_graph import plot_network

    # # a) Generate test data
    # data = Data(np.array([s1.returns, s2.returns, s3.returns, s4.returns]), dim_order="ps")

    # # b) Initialise analysis object and define settings
    # network_analysis = BivariateTE()
    # settings = dict(
    #     cmi_estimator='JidtGaussianCMI',
    #     max_lag_sources=8,
    #     min_lag_sources=1,
    #     verbose=True
    # )

    # # c) Run analysis
    # results = network_analysis.analyse_network(settings=settings, data=data)

    # # d) Plot inferred network to console and via matplotlib
    # results.print_edge_list(weights='max_te_lag', fdr=False)
    # plot_network(results=results, weights='max_te_lag', fdr=False)
    # plt.savefig(pathlib.Path.home() / 'TE.png')
    # plt.clf()



    # # Cross Correlation
    # from scipy import signal
    # corr = signal.correlate(s1.returns, s3.returns, mode='full')
    # print(len(s1.returns), len(corr))
    # x = signal.correlation_lags(len(s1.returns), len(s2.returns), mode='full')
    # plt.plot(x, corr)
    # # plt.xlim(-100, 100)
    # print(f"Max cross correlation at {x[corr == max(corr)]} lag.")
    # plt.savefig(pathlib.Path.home() / 'cross_correlation.png')
    # plt.clf()


    def a(x):
        return (x[0]-x[2])*(x[0]+x[2])/(-(x[0]-x[2])*(x[2]**2-x[1]**2) + (x[2]-x[1])*(x[0]**2-x[2]**2))

    def b(x):
        return (x[0]-x[2])*(x[0]**2-x[2]**2)/(-(x[0]-x[2])*(x[2]**2-x[1]**2) + (x[2]-x[1])*(x[0]**2-x[2]**2))

    def c(x, aa, bb):
        return x[2] - aa*x[1]*x[1] - bb*x[1]

    x = [-11,2,19]
    aa=a(x) 
    bb=b(x)
    cc=c(x,aa,bb)
    print(f"y=({aa})x^2 + ({bb})x + {cc}")