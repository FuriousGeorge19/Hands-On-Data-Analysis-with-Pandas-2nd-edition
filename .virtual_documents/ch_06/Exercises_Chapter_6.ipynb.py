get_ipython().run_line_magic("matplotlib", " inline")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


fb = pd.read_csv(
    'data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)

quakes = pd.read_csv('data/earthquakes.csv')


quakes.head(3)


q2 = (quakes.query('magType == "mb"')
      .loc[:, ['mag', 'tsunami']]
     )


fig, ax = plt.subplots(figsize=(12,8))  ## THIS CONSTRUCTION IS WORTH A NOTE...TIED TO PASSING THE AX ARGUMENT TO THE PLOTTING FUNCTION

sns.heatmap(
    q2.corr(),
    annot=True,
    center = 0,
    vmin=-1,
    vmax=1,
    ax=ax, ############  THIS IS KEY TO USING THE figure AND ax CREATED ABOVE WITH THE figsize YOU WANT
    annot_kws={'size': 26},
    cmap='PiYG'
)

plt.title('Correlation Between Earthquake Magnitude and Occurrence\n of Tsunamis for Earthquakes Measured w/Magnitude Type "mb" \n',
          fontdict = {'fontsize' : 20})

plt.xticks(size = 18)
plt.yticks(size = 18)



### plt.rcParams.keys() # USEFUL FOR SEEING THE PARAMETER OPTIONS FOR MATPLOTLIB





fb[['volume', 'close']].quantile([.25, .50, .75])


fb[['volume', 'close']].plot(kind='box', subplots=True, layout=(1,2), figsize=(12,6))


fb[['volume', 'close']].plot(kind='box', subplots=True, layout=(2,1), figsize=(10,8), vert=False)






























