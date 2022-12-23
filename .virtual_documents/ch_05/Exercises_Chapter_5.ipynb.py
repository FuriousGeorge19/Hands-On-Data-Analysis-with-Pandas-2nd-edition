import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


get_ipython().run_line_magic("matplotlib", " inline")


fb = pd.read_csv('data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True)
fb.head(3)



(fb.loc[:, ['open', 'high']]
 .rolling('20D')
 .min()
 .plot(figsize=(10,6),
      title = "20 Day Rolling Minimum Price")
)


# me, added 'high' to the plot not rolling to see if rolling values made sense.
# they seem to
(fb.loc[:, ['open', 'high']]
 .assign(
     open20dmin = lambda x: x.open.rolling('20D').min(),
     high20dmin = lambda x: x.high.rolling('20D').min())
 .drop(columns='open')
 .plot(figsize=(10,6))
 )


fb_chg=\
(fb
 .assign(chg = lambda x: x.close - x.open)
 .loc[:, ['chg']]
)

# histogram of chg
ax = fb_chg.chg.plot(figsize=(10,6), kind='hist', alpha=0.3, density=True, bins=20)

# kernel density plotted over histogram, using ax=ax to use the histogram assigned to ax in prior step
fb_chg.chg.plot(ax=ax, kind='kde',color='blue')
plt.xlabel('Daily Change ($) from Open to Close')



eq = pd.read_csv('data/earthquakes.csv')
eq.head(3)


(eq
 .assign(time = lambda x: pd.to_datetime(x.time, unit='ms')) # convert time object to datetime
 .set_index('time')
).head(3)


(eq
 .assign(time = lambda x: pd.to_datetime(x.time, unit='ms')) # convert time object to datetime
 .set_index('time')
 .sort_index()
 .query('parsed_place == "Indonesia"')
 .loc[:, ['magType', 'mag']]
 .groupby('magType')
 .boxplot(subplots=False) # I GUESS THE DEFAULT BEHAVIOR FOR PLOTTING GROUPBYS IS TO PLOT EACH
                          # ON A SEPARATE PLOT. SO `subplots=False` SEEMS TO BE REQUIRED TO PLOT
                          # THEM ON THE SAME CHART
)

plt.title("Boxplots of Earthquakes in Indonesia by magType")
plt.xlabel("magType")
plt.ylabel("Magnitude")


(
    fb
    .resample('1W')
    .agg({'high':'max', 'low':'min'})
    .assign(spread = lambda x: x.high.sub(x.low))[['spread']]
    .plot(title="FB: Weekly High Price - Weekly Low Price",
          ylabel="Price",
          figsize=(10,5)
         )
    
)


c19 = pd.read_csv('data/covid19_cases.csv', index_col= 'dateRep', parse_dates=True, dayfirst=True)
c19.head()


#c19.info()


countries = ['Brazil', 'China', 'India', 'Italy', 'Spain', 'USA']

(
    c19
    .replace({'United_States_of_America' : "USA"})
    .pivot_table(index='dateRep', columns = 'countriesAndTerritories', values = 'cases')
    .loc[:, countries]
    .sort_index()
    .diff()
    .rolling('14d').mean()
    .plot(figsize=(15,10))

)


c19_pivot =\
(c19
 .replace({'United_States_of_America' : "USA"})
 .pivot_table(index='dateRep', columns = 'countriesAndTerritories', values = 'cases')
 .loc[:, countries]
 .sort_index()
 .diff()
 .rolling('14d')
 .mean()
)


fig, axes = plt.subplots(1, 3, figsize=(15,5))

(c19_pivot[['China']]
 .plot(ax=axes[0], style='-.c', linewidth=1.0)
)

(
    c19_pivot[['Spain', 'Italy']]
    .plot(ax=axes[1], style=['--', ':'])
)

(
    c19_pivot[['Brazil', 'India', 'USA']]
    .plot(ax=axes[2], style=['r--', 'g:','k-'])
)





fb.head()





fig, axes = plt.subplots(1, 2, figsize=(15,5))

fb2=\
(
    fb
    .assign(diff_d = lambda x: x.open - x.close.shift(1))
    .loc[:, ['diff_d']]
)

fb2.plot(ax=axes[0], title = "Daily Pre-Market Change in FB Stock Price")


fb2.head()


import matplotlib.dates as mdates
import matplotlib.cbook as cbook

fig, axes = plt.subplots(1, 2, figsize=(15,5))

fb2.plot(ax=axes[0], title = "Daily Pre-Market Change in FB Stock Price")

fb3=\
(
    fb2
    .resample('1M')
    .sum()
)

fb3.index = fb3.index.strftime('get_ipython().run_line_magic("b-%y')", "")
fb3.plot(kind='bar', ax=axes[1], title="Cumulative Monthly Change in FB Stock Price")

plt.xticks(rotation=45)






(fb3.diff_d > 0).map({True: 'g',False: 'r'}).values





import matplotlib.dates as mdates
import matplotlib.cbook as cbook

fig, axes = plt.subplots(1, 2, figsize=(15,5))

fb2.plot(ax=axes[0], title = "Daily Pre-Market Change in FB Stock Price")

fb3=\
(
    fb2
    .resample('1M')
    .sum()
)

fb3.index = fb3.index.strftime('get_ipython().run_line_magic("b-%y')", "")


axes[1] =fb3.plot(y=0, kind='bar', ax=axes[1],\
                  title="Cumulative Monthly Change in FB Stock Price", \
                  color=['r', 'g', 'r', 'g', 'r', 'r', 'r', 'g', 'r', 'g', 'r', 'r']) ### <<<< LOOK HERE


plt.xticks(rotation=45)


import matplotlib.dates as mdates
import matplotlib.cbook as cbook

fig, axes = plt.subplots(1, 2, figsize=(15,5))

fb2.plot(ax=axes[0], title = "Daily Pre-Market Change in FB Stock Price")

fb3=\
(
    fb2
    .resample('1M')
    .sum()
)

fb3.index = fb3.index.strftime('get_ipython().run_line_magic("b-%y')", "")


axes[1] =fb3.plot(y=0, kind='bar', ax=axes[1],\
                  title="Cumulative Monthly Change in FB Stock Price", \
                  color=(fb3.diff_d > 0).map({True: 'g',False: 'r'})) ### <<<< LOOK HERE


plt.xticks(rotation=45)



























