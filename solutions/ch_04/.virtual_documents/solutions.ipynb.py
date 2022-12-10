import pandas as pd
import numpy as np

quakes = pd.read_csv('../../ch_04/exercises/earthquakes.csv')
faang = pd.read_csv('../../ch_04/exercises/faang.csv', index_col='date', parse_dates=True)


quakes.query(
    "parsed_place == 'Japan' and magType == 'mb' and mag >= 4.9"
)[['mag', 'magType', 'place']]


quakes.query("magType == 'ml'").assign(
    mag_bin=lambda x: pd.cut(x.mag, np.arange(0, 10))
).mag_bin.value_counts()


faang.groupby('ticker').resample('1M').agg(
    {
        'open': np.mean,
        'high': np.max,
        'low': np.min,
        'close': np.mean,
        'volume': np.sum
    }
)


pd.crosstab(quakes.tsunami, quakes.magType, values=quakes.mag, aggfunc='max')


faang.groupby('ticker').rolling('60D').agg(
    {
        'open': np.mean,
        'high': np.max,
        'low': np.min,
        'close': np.mean,
        'volume': np.sum
    }
)


faang.pivot_table(index='ticker')


faang.loc['2018-Q4'].query("ticker == 'AMZN'").drop(columns='ticker').apply(
    lambda x: x.sub(x.mean()).div(x.std())
).head()


events = pd.DataFrame({
    'ticker': 'FB',
    'date': pd.to_datetime(
         ['2018-07-25', '2018-03-19', '2018-03-20']
    ), 
    'event': [
         'Disappointing user growth announced after close.',
         'Cambridge Analytica story',
         'FTC investigation'
    ]
}).set_index(['date', 'ticker'])

faang.reset_index().set_index(['date', 'ticker']).join(
    events, how='outer'
).sample(10, random_state=0)


faang = faang.reset_index().set_index(['ticker', 'date'])
faang_index = (faang / faang.groupby(level='ticker').transform('first'))

# view 3 rows of the result per ticker
faang_index.groupby(level='ticker').agg('head', 3)


covid = pd.read_csv('../../ch_04/exercises/covid19_cases.csv')\
    .assign(date=lambda x: pd.to_datetime(x.dateRep, format='get_ipython().run_line_magic("d/%m/%Y'))\", "")
    .set_index('date')\
    .replace('United_States_of_America', 'USA')\
    .replace('United_Kingdom', 'UK')\
    .sort_index()


top_five_countries = covid\
    .groupby('countriesAndTerritories').cases.sum()\
    .nlargest(5).index

covid[covid.countriesAndTerritories.isin(top_five_countries)]\
    .groupby('countriesAndTerritories').cases.idxmax()


covid\
    .groupby(['countriesAndTerritories', pd.Grouper(freq='1D')]).cases.sum()\
    .unstack(0).diff().rolling(7).mean().last('1W')[top_five_countries]


covid\
    .pivot(columns='countriesAndTerritories', values='cases')\
    .drop(columns='China')\
    .apply(lambda x: x[x > 0].index.min())\
    .sort_values()\
    .rename(lambda x: x.replace('_', ' '))


covid\
    .pivot_table(columns='countriesAndTerritories', values='cases', aggfunc='sum')\
    .T\
    .transform('rank', method='max', pct=True)\
    .sort_values('cases', ascending=False)\
    .rename(lambda x: x.replace('_', ' '))
