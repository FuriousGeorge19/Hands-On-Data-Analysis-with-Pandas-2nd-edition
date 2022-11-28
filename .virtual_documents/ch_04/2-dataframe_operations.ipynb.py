import numpy as np
import pandas as pd

weather = pd.read_csv('data/nyc_weather_2018.csv', parse_dates=['date'])
weather.head()


fb = pd.read_csv('data/fb_2018.csv', index_col='date', parse_dates=True)
fb.head()


# my try
fb.assign(
    vol_3z_abs = lambda x: x.volume.sub(x.volume.mean()).div(x.volume.std()).abs()
).query('vol_3z_abs > 3')


#my try
fb_temp = fb.assign(
    vol_z = fb.volume.sub(fb.volume.mean()).div(fb.volume.std())
)

fb_temp[fb_temp.vol_z.abs() > 3 ]





fb.assign(
    abs_z_score_volume=lambda x: \
        x.volume.sub(x.volume.mean()).div(x.volume.std()).abs()
).query('abs_z_score_volume > 3')


fb.volume.pct_change().abs().rank(ascending=False).sort_values()


fb.assign(
    vol_abs_z=lambda x: \
        x.volume.sub(x.volume.mean()).div(x.volume.std()).abs(),
    vol_pct_chg = lambda x: x.volume.pct_change(),
    vol_pct_chg_rank = lambda x : x.vol_pct_chg.abs().rank(ascending=False)
)\
.nsmallest(5, 'vol_pct_chg_rank' )





# my try
fb.assign(
    vol_pct_abs = lambda x:\
    x.volume.pct_change().abs(),
    vol_pct_abs_rank = lambda x:\
    x.vol_pct_abs.rank(ascending=False)
).sort_values(by='vol_pct_abs_rank', ascending=True).head(5)






fb.assign(
    volume_pct_change=fb.volume.pct_change(),
    pct_change_rank=lambda x: \
        x.volume_pct_change.abs().rank(ascending=False)
).nsmallest(5, 'pct_change_rank')


fb['2018-01-11':'2018-01-12']


# me
fb['2018-Q1':'2018-Q1'].head()


# me
fb.loc['2018-Q1':'2018-Q1', :].head()


# me
(fb.low > 215).sum()


# me
(fb.low > 215).any()





(fb > 215).any()


# me
# logic -> All had at least one day they were less than or equal to 215 equivalent to
# All of them > 215 on every day is False

(fb[['open', 'high', 'low', 'close']] > 215).all()





(fb > 215).all()


pd.cut??


# me
pd.cut(fb.volume, bins=3, labels=['low', 'med','high']).value_counts()


# me
(fb.volume.value_counts() > 1).sum()





(fb.volume.value_counts() > 1).sum()


# me
pd.cut(fb.volume, bins=3, labels=[ 'low', 'medium', 'high'],retbins=True)


volume_binned = pd.cut(fb.volume, bins=3, labels=['low', 'med', 'high'])
volume_binned.value_counts()


# me
fb[volume_binned == 'high'].sort_values(by='volume', ascending=False)


fb.loc[volume_binned == 'high'].sort_values(by='volume', ascending=False)











fb[volume_binned == 'high'].sort_values(by='volume', ascending=False)


fb['2018-07-25':'2018-07-27']





fb[volume_binned == 'high'].sort_values('volume', ascending=False)


fb['2018-07-25':'2018-07-26']


fb['2018-03-16':'2018-03-20']


get_ipython().getoutput("cp -r ../visual-aids/visual_aids .")


from visual_aids.misc_viz import low_med_high_bins_viz

low_med_high_bins_viz(
    fb, 'volume', ylabel='volume traded',
    title='Daily Volume Traded of Facebook Stock in 2018 (with bins)'
)


#pd.qcut??


#me
vol_qbin = pd.qcut(fb.volume, q=4, labels=['q1', 'q2','q3', 'q4'])


vol_qbin.value_counts()





# me
vol_qbin = pd.qcut(fb.volume, q=4, labels = ['q1', 'q2', 'q3', 'q4'])
vol_qbin.value_counts()








volume_qbinned = pd.qcut(fb.volume, q=4, labels=['q1', 'q2', 'q3', 'q4'])
volume_qbinned.value_counts()


from visual_aids.misc_viz import quartile_bins_viz

quartile_bins_viz(
    fb, 'volume', ylabel='volume traded', 
    title='Daily Volume Traded of Facebook Stock in 2018 (with quartile bins)'
)


#me
weather.head()


# cp_weather = (
#     weather.pivot(columns='datatype', values='value')
#     
# )

cp_weather = (
    weather.query('station == "GHCND:USW00094728"')
    .pivot(index=['date', 'station'], columns='datatype', values='value')
)

cp_weather.head()


#pd.pivot?








central_park_weather = weather\
    .query('station == "GHCND:USW00094728"')\
    .pivot(index='date', columns='datatype', values='value')
central_park_weather.head()











central_park_weather_z = (
    central_park_weather
    .loc['2018-10' ,['TMIN', 'TMAX', 'PRCP']]
    .apply(
        lambda x: x.sub(x.mean()).div(x.std())
    )
)

central_park_weather_z.head()








# me
central_park_weather_temp = central_park_weather.loc['2018-10', :].assign(
    TMIN_z = lambda x: x.TMIN.sub(x.TMIN.mean()).div(x.TMIN.std()), 
    TMAX_z = lambda x: x.TMAX.sub(x.TMAX.mean()).div(x.TMAX.std()),
    PRCP_z =lambda x: x.PRCP.sub(x.PRCP.mean()).div(x.PRCP.std()),
).sort_index(axis=1)
central_park_weather_temp.head()


# my first approach didn't use apply(), let me try that after glancing at the answer below
central_park_weather_temp = \
central_park_weather.loc['2018-10', ['TMIN', 'TMAX', 'PRCP' ]].apply(
    lambda x: x.sub(x.mean()).div(x.std())
)

central_park_weather_temp.head()





oct_weather_z_scores = central_park_weather\
    .loc['2018-10', ['TMIN', 'TMAX', 'PRCP']]\
    .apply(lambda x: x.sub(x.mean()).div(x.std()))
oct_weather_z_scores.describe().T


oct_weather_z_scores = central_park_weather\
    .loc['2018-10', ['TMIN', 'TMAX', 'PRCP']]\
    .apply(lambda x: x.sub(x.mean()).div(x.std()))


cp_z = (
    central_park_weather.loc['2018-10', ['TMIN', 'TMAX', 'PRCP' ]]
    .apply(lambda x: x.sub(x.mean()).div(x.std()))
    .rename(columns = lambda x: x + '_z')
)
cp_z.head()



(
    central_park_weather
    .loc['2018-10', ['TMIN', 'TMAX', 'PRCP' ]]
    .join(cp_z, how='right')
    .sort_index(axis=1)
).head()














# me
oct_weather_z_scores.query('PRCP > 3').PRCP


oct_weather_z_scores.query('PRCP > 3').PRCP


central_park_weather.loc['2018-10', 'PRCP'].describe()


# me. I thought whole numbers just meant the integer piece. 
fb.astype(int).astype(str).apply(
    lambda x: x.str.len()
).sum().sum()


fb.head()


fb.apply(
    lambda x: np.vectorize(lambda y: len(str(np.ceil(y))))(x)
).astype('int64').head()


fb.apply(
    lambda x: np.vectorize(lambda y: len(str(np.ceil(y))))(x)
).astype('int64').equals(
    fb.applymap(lambda x: len(str(np.ceil(x))))
)


import time

import numpy as np
import pandas as pd

np.random.seed(0)

vectorized_results = {}
iteritems_results = {}

for size in [10, 100, 1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]:
    # set of numbers to use
    test = pd.Series(np.random.uniform(size=size))
    
    # time the vectorized operation
    start = time.time()
    x = test + 10
    end = time.time()
    vectorized_results[size] = end - start
    
    # time the operation with `iteritems()`
    start = time.time()
    x = []
    for i, v in test.iteritems():
        x.append(v + 10)
    x = pd.Series(x)
    end = time.time()
    iteritems_results[size] = end - start

results = pd.DataFrame(
    [pd.Series(vectorized_results, name='vectorized'), pd.Series(iteritems_results, name='iteritems')]
).T    

# plotting
ax = results.plot(title='Time Complexity', color=['blue', 'red'], legend=False)

# formatting
ax.set(xlabel='item size (rows)', ylabel='time (s)')
ax.text(0.5e7, iteritems_results[0.5e7] * .9, 'iteritems()', rotation=34, color='red', fontsize=12, ha='center', va='bottom')
ax.text(0.5e7, vectorized_results[0.5e7], 'vectorized', color='blue', fontsize=12, ha='center', va='bottom')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)






# me
central_park_weather_temp = central_park_weather.assign(
    PRCP_3dsum = lambda x: x.PRCP.rolling(3).sum()
).sort_index(axis=1)
central_park_weather_temp.head(10)


central_park_weather.loc['2018-10'].assign(
    rolling_PRCP=lambda x: x.PRCP.rolling('3D').sum()
)[['PRCP', 'rolling_PRCP']].head(7).T


# me
central_park_weather.loc['2018-10'].rolling('3D').mean().head(7).iloc[:, :6]











central_park_weather.loc['2018-10'].rolling('3D').mean().head(7).iloc[:,:6]


central_park_weather['2018-10-01':'2018-10-07'].rolling('3D').agg(
    {'TMAX': 'max', 'TMIN': 'min', 'AWND': 'mean', 'PRCP': 'sum'}
).join( # join with original data for comparison
    central_park_weather[['TMAX', 'TMIN', 'AWND', 'PRCP']], 
    lsuffix='_rolling'
).sort_index(axis=1) # sort columns so rolling calcs are next to originals


fb_reindexed = fb\
    .reindex(pd.date_range('2018-01-01', '2018-12-31', freq='D'))\
    .assign(
        volume=lambda x: x.volume.fillna(0),
        close=lambda x: x.close.fillna(method='ffill'),
        open=lambda x: x.open.combine_first(x.close),
        high=lambda x: x.high.combine_first(x.close),
        low=lambda x: x.low.combine_first(x.close)
    )
fb_reindexed.assign(day=lambda x: x.index.day_name()).head(10)


fb.head(5)


fb.reindex(pd.date_range('2018-01-01', '2018-12-31', freq='D')).head(15)


# pd.Series.combine_first??


from pandas.api.indexers import VariableOffsetWindowIndexer

indexer = VariableOffsetWindowIndexer(
    index=fb_reindexed.index, offset=pd.offsets.BDay(3)
)
fb_reindexed.assign(window_start_day=0).rolling(indexer).agg({
    'window_start_day': lambda x: x.index.min().timestamp(),
    'open': 'mean', 'high': 'max', 'low': 'min',
    'close': 'mean', 'volume': 'sum'
}).join(
    fb_reindexed, lsuffix='_rolling'
).sort_index(axis=1).assign(
    day=lambda x: x.index.day_name(),
    window_start_day=lambda x: pd.to_datetime(x.window_start_day, unit='s')
).head(10)





central_park_weather.head()


#me
(central_park_weather
 .loc['2018-10', ['PRCP']]
 .expanding().mean()
).head(10)








central_park_weather.loc['2018-06'].assign(
    TOTAL_PRCP=lambda x: x.PRCP.cumsum(),
    AVG_PRCP=lambda x: x.PRCP.expanding().mean()
).head(10)[['PRCP', 'TOTAL_PRCP', 'AVG_PRCP']].T


central_park_weather['2018-10-01':'2018-10-07'].expanding().agg(
    {'TMAX': np.max, 'TMIN': np.min, 'AWND': np.mean, 'PRCP': np.sum}
).join(
    central_park_weather[['TMAX', 'TMIN', 'AWND', 'PRCP']], 
    lsuffix='_expanding'
).sort_index(axis=1)


central_park_weather.assign(
    AVG=lambda x: x.TMAX.rolling('30D').mean(),
    EWMA=lambda x: x.TMAX.ewm(span=30).mean()
).loc['2018-09-29':'2018-10-08', ['TMAX', 'EWMA', 'AVG']].T


# me do the above w/out looking
# (
#     data.pipe(h)
#     .pipe(g, 20)
#     .pipe(f, x=True)
# )
 











def get_info(df):
    return 'get_ipython().run_line_magic("d", " rows, %d columns and max closing Z-score was %d' % (*df.shape, df.close.max())")

# the part before '==' shows an example w/out using pipe
get_info(fb.loc['2018-Q1'].apply(lambda x: (x - x.mean())/x.std()))\
    == fb.loc['2018-Q1'].apply(lambda x: (x - x.mean())/x.std()).pipe(get_info)

# the part after the '==' makes use of the pipe and shows they are equivalent


# me
# I find this example confusing as fuck
# Let's work from the inside out
fb.loc['2018-Q1'].apply(lambda x: (x - x.mean())/x.std()).head()


# me..next step
get_info(fb.loc['2018-Q1'].apply(lambda x: (x - x.mean())/x.std()))





fb.pipe(pd.DataFrame.rolling, '20D').mean().equals(fb.rolling('20D').mean())


pd.DataFrame.rolling(fb, '20D').mean().equals(fb.rolling('20D').mean())


from window_calc import window_calc
window_calc??


window_calc(fb, pd.DataFrame.expanding, np.median).head()


window_calc(fb, pd.DataFrame.ewm, 'mean', span=3).head()


window_calc(
    central_park_weather.loc['2018-10'], 
    pd.DataFrame.rolling, 
    {'TMAX': 'max', 'TMIN': 'min', 'AWND': 'mean', 'PRCP': 'sum'},
    '3D'
).head()
