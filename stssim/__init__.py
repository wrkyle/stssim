from scipy import stats
from scipy import signal
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
        Matplotlib.pyplot.Figure
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

def comparison_plot(series1, series2):

    fig = plt.figure(constrained_layout=True, figsize=(6, 9.6), dpi=400)
    gs = fig.add_gridspec(nrows=4, ncols=2)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(series1)
    ax1.set_title('Series 1')
    ax1.tick_params(right=True, labelright=True)
    ax1.annotate(
        f"n={len(series1)}",
        xy=(.03, .95),
        xycoords='axes fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(series2)
    ax2.set_title('Series 2')
    ax2.tick_params(right=True, labelright=True)
    ax2.annotate(
        f"n={len(series2)}",
        xy=(.03, .95),
        xycoords='axes fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )

    # Pearson R
    returns1 = np.insert(np.diff(series1), 0, 0)
    returns2 = np.insert(np.diff(series2), 0, 0)
    r, p_value = stats.pearsonr(returns1, returns2)
    
    # Cross correlation to find lag.
    corr = signal.correlate(returns1, returns2, mode='full')
    x = signal.correlation_lags(len(returns1), len(returns2), mode='full')
    lag = x[corr == max(corr)][0]
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(x, corr)
    ax3.set_title('Cross Correlation')
    ax3.annotate(
        "max($\\rho$) @ $\ell$=" + f"{lag}",
        xy=(.03, .95),
        xycoords='axes fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )

    # Pearson R redone with lag removed.
    if lag < 0:
        r1 = returns1[:lag]
        r2 = returns2[-lag:]
        r_lag, p_value_lag = stats.pearsonr(r1, r2)
    elif lag > 0:
        r1 = returns1[lag:]
        r2 = returns2[:-lag]
        r_lag, p_value_lag = stats.pearsonr(r1, r2)

    # Scatterplot
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.scatter(returns1, returns2, s=2)
    ax4.set_title('Scatterplot')
    ax4.set_xlabel("series 1 returns")
    ax4.set_ylabel("series 2 returns")

    # Textual summary
    ax5 = fig.add_subplot(gs[3, 1])
    annotation = [
        "$\\rho$=" + f"{r}",
        f"p-value={p_value}",
        "$\\rho_{\ell}$=" + f"{r_lag}",
        "p-value$_{\ell}$=" + f"{p_value_lag}",
    ]
    ax5.annotate(
        "\n".join(annotation),
        xy=(.5, .5),
        xycoords='axes fraction',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=10,
        bbox=dict(boxstyle='square,pad=.05', fc='#ffffff99', ec='none')
    )
    ax5.set_axis_off()

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

    def compare_with(self, series, filepath):
        """

        Plot a scatterplot with another series to see how returns are related.
    
        Args:
            series (Series): Another time series.
            filepath (str): The filepath to save the resulting figure to.
    
        Returns:
            None: The return value.
        """
    
        plt.clf()
        f = comparison_plot(self.series, series.series)
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

    pass