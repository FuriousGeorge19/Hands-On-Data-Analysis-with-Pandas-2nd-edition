get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fb = pd.read_csv(
    'data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)


from pandas.plotting import scatter_matrix
scatter_matrix(fb, figsize=(10, 10))


scatter_matrix(fb, figsize=(10, 10), diagonal='kde')


from pandas.plotting import lag_plot
np.random.seed(0) # make this repeatable
lag_plot(pd.Series(np.random.random(size=200)))


lag_plot(fb.close)


lag_plot(fb.close, lag=5)


from pandas.plotting import autocorrelation_plot
np.random.seed(0) # make this repeatable
autocorrelation_plot(pd.Series(np.random.random(size=200)))


autocorrelation_plot(fb.close)


from pandas.plotting import bootstrap_plot
fig = bootstrap_plot(fb.volume, fig=plt.figure(figsize=(10, 6)))
