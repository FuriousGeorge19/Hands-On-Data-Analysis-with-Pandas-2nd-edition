import numpy as np
import pandas as pd

fb = pd.read_csv('data/fb_2018.csv', index_col='date', parse_dates=True).assign(
    trading_volume=lambda x: pd.cut(x.volume, bins=3, labels=['low', 'med', 'high'])
)
fb.head()


weather = pd.read_csv('data/weather_by_station.csv', index_col='date', parse_dates=True)
weather.head()


weather.shape


pd.set_option('display.float_format', lambda x: 'get_ipython().run_line_magic(".2f'", " % x)")





fb.head()


#me
fb.agg({'open': ['min', 'max'], 'high':['median'], 'low':'mean'})





# me
fb.agg({'open': 'mean', 'close':['min', 'mean', 'sum']})


fb.agg({
    'open': np.mean, 
    'high': np.max, 
    'low': np.min, 
    'close': np.mean, 
    'volume': np.sum
})


# me
weather.head()


# me
(
    weather
    .query('station == "GHCND:USW00094728"')
    .pivot(columns='datatype', values='value')
    .loc['2018', ['PRCP', 'SNOW']]
    .agg(sum)
)





# me
weather.query("datatype in ('SNOW', 'PRCP') and station == 'GHCND:USW00094728'") \
.pivot_table(index = 'date', values = 'value', columns = 'datatype')\
.agg(sum)


#me
weather.query("datatype in ('SNOW', 'PRCP')") \
.query('station == "GHCND:USW00094728"') \
.pivot(columns = 'datatype', values = 'value').head()


#me
weather.query("datatype in ('SNOW', 'PRCP')") \
.query('station == "GHCND:USW00094728"') \
.pivot(columns = 'datatype', values = 'value') \
.agg(sum)


# me
weather.query("datatype in ('SNOW', 'PRCP')") \
.query('station == "GHCND:USW00094728"') \
.pivot_table(values = 'value', columns = 'datatype', aggfunc=sum)



weather.query('station == "GHCND:USW00094728"')\
    .pivot(columns='datatype', values='value')[['SNOW', 'PRCP']]\
    .sum()


weather.query('station == "GHCND:USW00094728"')\
    .pivot(columns='datatype', values='value')[['SNOW', 'PRCP']]\
    .agg('sum')


fb.agg({
    'open': 'mean',
    'high': ['min', 'max'],
    'low': ['min', 'max'],
    'close': 'mean'
})


fb.head()


# me
(
    fb
    .groupby('trading_volume')
    .agg({'open': 'min', 'high':[np.mean, np.sum], 'low':'median'})
)


# me
fb.groupby('trading_volume').mean()


# me
fb.groupby('trading_volume').agg({'open':[np.min, np.max], 'close':np.sum, 'volume':[np.min, np.mean, np.sum] })





fb.groupby('trading_volume').mean()


fb.groupby('trading_volume')['close'].agg(['min', 'max', 'mean'])


fb_agg = fb.groupby('trading_volume').agg({
    'open': 'mean',
    'high': ['min', 'max'],
    'low': ['min', 'max'],
    'close': 'mean'
})
fb_agg


fb_agg.columns


new_columns = ['_'.join(x) for x in fb_agg.columns]
new_columns


fb_new = fb_agg.copy()
fb_new.columns = new_columns
fb_new














fb_agg.columns = ['_'.join(col_agg) for col_agg in fb_agg.columns]
fb_agg.head()


weather.head()


# me
weather.loc['2018-10'].query('datatype == "PRCP"')['value'] \
.groupby(level=0).mean().head(13)


# what does the above do? I think it averages PRCP per day, across stations. 

# start w/the first part of the code above:
weather.loc['2018-10'].query('datatype == "PRCP"').head()

# we see multiple stations reporting PRCP


# select the 'value' column and then group by level=0, which means the index, which in this case...is the date.
weather.loc['2018-10'].query('datatype == "PRCP"')['value']\
.groupby(level=0).mean().head(13)








weather.loc['2018-10'].query('datatype == "PRCP"')\
    .groupby(level=0).mean(numeric_only=True).head().squeeze()


# I THINK squeeze turns a single series dataframe into a series


type(weather.loc['2018-10'].query('datatype == "PRCP"')\
    .groupby(level=0).mean(numeric_only=True).head())


type(weather.loc['2018-10'].query('datatype == "PRCP"')\
    .groupby(level=0).mean(numeric_only=True).head().squeeze())


# YUP!


#me
weather.head()


# me
(
    weather
    .query('datatype == "PRCP"')
    .groupby([pd.Grouper(freq='Q'), 'station'])['value']
    .sum()
)


# me...adding unstack to the above might make it easier to read
(
    weather
    .query('datatype == "PRCP"')
    .groupby([pd.Grouper(freq='Q'), 'station'])['value']
    .sum()
    .unstack()
)


# me...reversing the order in groupby helps even more
(
    weather
    .query('datatype == "PRCP"')
    .groupby(['station', pd.Grouper(freq='Q')])['value']
    .sum()
    .unstack()
)





# me
weather.query('datatype == "PRCP"') \
.groupby( ['station', pd.Grouper(freq = 'Q')]).sum(numeric_only=True) \
.unstack().sample(5, random_state=1)


weather.query('datatype == "PRCP"').groupby(
    ['station', pd.Grouper(freq='Q')]
).sum(numeric_only=True).unstack().sample(5, random_state=1)


# me

weather.groupby('station_name') \
.filter( lambda x: x.name.endswith("NY US"))

# THE X IS THE ITEM GROUPED ON. IN THIS CASE, 'station_name'





weather.groupby('station_name').filter( # station names with "NY US" in them
    lambda x: x.name.endswith('NY US')
).query('datatype == "SNOW"').groupby('station_name').sum(numeric_only=True).squeeze() # aggregate and make a series (squeeze)


 # me ---> I'm curious what happens without the second groupby
weather.groupby('station_name').filter( # station names with "NY US" in them
    lambda x: x.name.endswith('NY US')
).query('datatype == "SNOW"').head()





weather.head()


(
    weather
    .query('datatype == "PRCP"')
    #.groupby(pd.Grouper(freq='1D'))  --> These two versions of groupby are interchangeable
    .groupby(level=0)
    .mean(numeric_only=True)
    .groupby(pd.Grouper(freq='M'))
    .sum()
    .nlargest(columns='value', n=12)

)








# me
weather.query('datatype == "PRCP"') \
.rename({'value':'Monthly_PRCP'}, axis=1) \
.groupby(level=0).mean(numeric_only=True) \
.groupby(pd.Grouper(freq='M')).sum().sort_values(by = 'Monthly_PRCP', ascending = False)





weather.query('datatype == "PRCP"')\
    .groupby(level=0).mean()\
    .groupby(pd.Grouper(freq='M')).sum().value.nlargest()


weather.head()


# me
# This solution requires me to turn a series back into a dataframe . Seems inelegant. 

PRCP_by_day = (
    weather
    .query('datatype == "PRCP"')
    .groupby(level=0)['value']
    .mean()
    .rename('PRCP_Daily')
    .to_frame()
    .assign(
        PRCP_Monthly = lambda x: x.PRCP_Daily.groupby(pd.Grouper(freq='M')).transform(sum),
        PRCP_PTT = lambda x: x.PRCP_Daily.div(x.PRCP_Monthly).mul(100)
   
    )
    
)
PRCP_by_day.head(10)


PRCP_by_day.loc['2018-01'].sum()


# me
# This solution eliminates the need for to_frame by calling mean with numeric_only, so the groupby isn't turned into a series
    weather
    .query('datatype == "PRCP"')
    .groupby(level=0)
    .mean(numeric_only=True)
    .rename({'value':'PRCP_Daily'}, axis=1)
    .assign(
        PRCP_Monthly = lambda x: x.PRCP_Daily.groupby(pd.Grouper(freq='M')).transform(sum),
        PRCP_PTT = lambda x: x.PRCP_Daily.div(x.PRCP_Monthly).mul(100)
   
    )
)
     
    
PRCP_by_day.head(10)














# me
weather.query('datatype == "PRCP"') \
.rename({'value':'PRCP'}, axis=1) \
.groupby(level = 0).mean(numeric_only = True) \
.assign(
    PRCP_M = lambda x: x.groupby(pd.Grouper(freq = 'M')).transform(np.sum),
    PRCP_PCT = lambda x: x.PRCP.div(x.PRCP_M)
).nlargest(10, 'PRCP_PCT')





weather.query('datatype == "PRCP"')\
    .rename(dict(value='prcp'), axis=1)\
    .groupby(level=0).mean()\
    .groupby(pd.Grouper(freq='M'))\
    .transform(np.sum)['2018-01-28':'2018-02-03']


weather\
    .query('datatype == "PRCP"')\
    .rename(dict(value='prcp'), axis=1)\
    .groupby(level=0).mean()\
    .assign(
        total_prcp_in_month=lambda x: \
            x.groupby(pd.Grouper(freq='M')).transform(np.sum),
        pct_monthly_prcp=lambda x: \
            x.prcp.div(x.total_prcp_in_month)
    )\
    .nlargest(5, 'pct_monthly_prcp')


fb.head()


(
    fb[['open', 'high', 'low', 'close']]
    .transform(lambda x: x.sub(x.mean()).div(x.std()))
)





# me
fb[['open', 'high', 'low', 'close']] \
.transform(lambda x: x.sub(x.mean()).div(x.std()) )





fb[['open', 'high', 'low', 'close']]\
    .transform(lambda x: (x - x.mean()).div(x.std()))\
    .head()


fb.head()


# me
fb.pivot_table(columns='trading_volume')


# me. Above, same as
fb.pivot_table(columns='trading_volume', aggfunc='mean')









fb.pivot_table(columns='trading_volume')


#me
fb.pivot_table(index='trading_volume')


#me
fb.groupby('trading_volume').mean()





fb.pivot_table(index='trading_volume')


weather.head()


# this looks...had to work with because of the date

(
    weather
    .reset_index()
    .pivot_table(
        index = ['date', 'station', 'station_name'], 
        columns='datatype', 
        values='value')
                       
)


# add reset_index again to get the multi_index into columns
(
    weather
    .reset_index()
    .pivot_table(
        index = ['date', 'station', 'station_name'], 
        columns='datatype', 
        values='value')
    .reset_index()
                       
)





weather.reset_index().pivot_table(
    index=['date', 'station', 'station_name'], 
    columns='datatype', 
    values='value',
    aggfunc='median'
).reset_index().tail()


# me
weather.reset_index().head(3)


# me
weather.reset_index().pivot_table(
    index=['date', 'station', 'station_name'], 
    columns='datatype', 
    values='value',
    aggfunc='median'
).head(3)


# me
weather.reset_index().pivot_table(
    index=['date', 'station', 'station_name'], 
    columns='datatype', 
    values='value',
    aggfunc='median'
).reset_index().tail()











pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month'] # name the columns index
)


pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month'],
    normalize='columns'
)


pd.crosstab(
    index=fb.trading_volume,
    columns=fb.index.month,
    colnames=['month'],
    values=fb.close,
    aggfunc=np.mean
)


snow_data = weather.query('datatype == "SNOW"')
pd.crosstab(
    index=snow_data.station_name,
    columns=snow_data.index.month,
    colnames=['month'],
    values=snow_data.value,
    aggfunc=lambda x: (x > 0).sum(),
    margins=True, # show row and column subtotals
    margins_name='total observations of snow' # name the subtotals
)
