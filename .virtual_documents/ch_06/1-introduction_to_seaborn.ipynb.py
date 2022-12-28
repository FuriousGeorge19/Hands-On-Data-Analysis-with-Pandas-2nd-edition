get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

fb = pd.read_csv(
    'data/fb_stock_prices_2018.csv', index_col='date', parse_dates=True
)
quakes = pd.read_csv('data/earthquakes.csv')


# me
fb.head(3)


quakes.head(3)


quakes.assign(
    time=lambda x: pd.to_datetime(x.time, unit='ms')
).set_index('time').loc['2018-09-28'].query(
    'parsed_place == "Indonesia" and tsunami and mag == 7.5'
)


#me

plt.subplots(figsize = (10,5))
sns.stripplot(
    x = 'magType',
    y = 'mag',
    data = quakes,
    hue = 'tsunami'
)



#?sns.stripplot





sns.stripplot(
    x='magType',
    y='mag',
    hue='tsunami',
    data=quakes.query('parsed_place == "Indonesia"')
)


#me
plt.subplots(figsize=(10,5))
sns.swarmplot(
    x = 'magType',
    y = 'mag',
    hue = 'tsunami',
    data = quakes.query('parsed_place == "Indonesia"'),
    size=5
)


#?sns.swarmplot


sns.swarmplot(
    x='magType',
    y='mag',
    hue='tsunami',
    data=quakes.query('parsed_place == "Indonesia"'),
    size=3.5 # point size
)


# me
plt.subplots(figsize=(10,5))
sns.boxenplot(
    x='magType',
    y='mag',
    #hue='tsunami',
    #data=quakes.query('parsed_place == "Indonesia"')
    data=quakes
    
)


#?sns.boxenplot





sns.boxenplot(
    x='magType', y='mag', data=quakes[['magType', 'mag']]
)
plt.title('Comparing earthquake magnitude by magType')


# me
fig, axes = plt.subplots(figsize=(10, 5))

sns.violinplot(
    x='magType', y='mag', data=quakes,  
    ax=axes, scale='width' # all violins have same width
)
plt.title('Comparing earthquake magnitude by magType')





fig, axes = plt.subplots(figsize=(10, 5))

sns.violinplot(
    x='magType', y='mag', data=quakes[['magType', 'mag']],  
    ax=axes, scale='width' # all violins have same width
)
plt.title('Comparing earthquake magnitude by magType')


sns.heatmap(
    fb.sort_index().assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    ).corr(),
    annot=True, center=0, vmin=-1, vmax=1
)


# me, play with colormap
fig, axes = plt.subplots(figsize=(8, 8))


sns.heatmap(
    fb.sort_index().assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    ).corr(),
    annot=True, center=0, vmin=-1, vmax=1,
    cmap = 'Blues' # <<< CHANGES COLORMAP
)


sns.pairplot(fb)


sns.pairplot(
    fb.assign(quarter=lambda x: x.index.quarter),
    diag_kind='kde',
    hue='quarter'
)


# me , use palette argument to modify appearance
sns.pairplot(
    fb.assign(quarter=lambda x: x.index.quarter),
    diag_kind='kde',
    hue='quarter', palette = 'hls',
)


sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)


sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='hex',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)


sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='kde',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)


sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='reg',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)


sns.jointplot(
    x='log_volume',
    y='max_abs_change',
    kind='resid',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low
    )
)
# update y-axis label (discussed in the next notebook)
plt.ylabel('residuals')


fb_reg_data = fb.assign(
    log_volume=np.log(fb.volume),
    max_abs_change=fb.high - fb.low
).iloc[:,-2:]


fb_reg_data.head()


import itertools


iterator = itertools.repeat("I'm an iterator", 3)

for i in iterator:
    print(f'-->{i}')
print('This printed once because the iterator has been exhausted')
for i in iterator:
    print(f'-->{i}')


iterable = list(itertools.repeat("I'm an iterable", 3))

for i in iterable:
    print(f'-->{i}')
print('This prints again because it\'s an iterable:')
for i in iterable:
    print(f'-->{i}')


from viz import reg_resid_plots
reg_resid_plots??


from viz import reg_resid_plots
reg_resid_plots(fb_reg_data)


# me

fb_reg_data = fb.assign(
    log_volume=np.log(fb.volume),
    max_abs_change=fb.high - fb.low
).iloc[:,[-4, -2, -1]]

fb_reg_data.head()


# me --> Tried changing figsize...didn't work.

reg_resid_plots(fb_reg_data)
plt.figure(figsize=(30,10))





sns.lmplot(
    x='log_volume',
    y='max_abs_change',
    data=fb.assign(
        log_volume=np.log(fb.volume),
        max_abs_change=fb.high - fb.low,
        quarter=lambda x: x.index.quarter
    ),
    col='quarter'
)


get_ipython().getoutput("pip install -U seaborn")


#?sns.FacetGrid


g = sns.FacetGrid(
    quakes.query(
        'parsed_place.isin(["Indonesia", "Papua New Guinea"]) '
        'and magType == "mb"'
    ),
    row='tsunami',
    col='parsed_place',
    height=4
)

# g = g.map(sns.histplot, 'mag', kde=True)   ## <<< ORIGINAL
g.map(sns.histplot, 'mag', kde=True)  



