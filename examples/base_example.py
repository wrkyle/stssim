import stssim
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Set random seed for reproducibility.
np.random.seed(seed=23342379)
N = 10000

# Some example distributions.
t = stats.nct(nc=.02, df=2.7)
norm = stats.norm(loc=0.1, scale=20)
uniform = stats.uniform(loc=-50, scale=100)



# Create a non-negative time series of N steps
s1 = stssim.StochasticSeries(N, distribution=t, x0=1, no_negative=True)

# Plot the time series.
s1.plot(pathlib.Path.home() / 'time_series.png', return_type='absolute')


# Create a new series linearly dependent on the first.
s2 = stssim.Series()
lag = 4
s2.from_dependencies([s1], generator=lambda dependencies, i: 2 * (0 if i < lag else dependencies[0].returns[i - lag]) + 2*(np.random.rand() - .5), generates="returns", x0=1)
s2.plot(pathlib.Path.home() / 'time_series2.png', return_type='absolute')
print(f"Linearly dependent on series 1 with a time lag of {lag}.")

# Examine Pearson correlation coefficient.
r = stats.pearsonr(s1.returns, s2.returns)
print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}\n\n")


# Create a new series non-linearly dependent on the first.
lag = 7
s3 = stssim.Series()
s3.from_dependencies([s1], generator=lambda dependencies, i: 0.0 if i < lag else np.sin(3 * dependencies[0].returns[i - lag]), generates="returns", x0=100)
s3.plot(pathlib.Path.home() / 'time_series3.png', return_type='absolute')
print(f"Non-linearly dependent on series 1 with a time lag of {lag}.")

# Examine Pearson correlation coefficient.
r = stats.pearsonr(s1.returns, s3.returns)
print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}\n\n")


# Create a new series with no dependencies.
s4 = stssim.StochasticSeries(N, distribution=norm, x0=100, no_negative=True)
s4.plot(pathlib.Path.home() / 'time_series4.png', return_type='absolute')

# Examine Pearson correlation coefficient.
r = stats.pearsonr(s1.returns, s4.returns)
print(f"Correlation Coefficient={r[0]}\nTwo-tailed p-value={r[1]}")


# Plot comparisons.
s1.compare_with(s2, pathlib.Path.home() / '1_2_compare.png')
s1.compare_with(s3, pathlib.Path.home() / '1_3_compare.png')
s1.compare_with(s4, pathlib.Path.home() / '1_4_compare.png')



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
