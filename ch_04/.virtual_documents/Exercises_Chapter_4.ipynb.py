import pandas as pd
import numpy as np


get_ipython().getoutput("ls exercises/")


get_ipython().getoutput("head -5 exercises/earthquakes.csv")


quakes = pd.read_csv('exercises/earthquakes.csv')


quakes.head()


quakes.dtypes


quakes.shape


quakes.loc[lambda x: (x.mag >= 4.9) & (x.magType == 'mb') ]


quakes[quakes.mag.ge(4.9) & (quakes.magType == 'mb')]


quakes.query('(mag >= 4.9) and (magType == "mb")')


##??pd.cut


ml_mags = (
    quakes
    .query('magType == "ml"')['mag']
    .to_frame()
)

# find min and max values
# agg might not work on a series
ml_mags.agg({'mag':['min', 'max']})


ml_mag_bins = pd.cut(ml_mags.squeeze(), bins = range(-2,7))
ml_mag_bins.head()


ml_mag_bins.value_counts()


ml_mag_bins.value_counts(normalize=True)


get_ipython().getoutput("head -5 exercises/faang.csv")


faang = pd.read_csv('exercises/faang.csv')
faang.head()


faang.shape


faang.info() # to group by month, want to make date the index as a datetime


faang = pd.read_csv('exercises/faang.csv', index_col='date', parse_dates=True)
faang.head()


faang.info()


( faang
 .groupby('ticker')
 .resample('M')
 .agg({
     'open':'mean',
     'high':'max',
     'low':'min',
     'close':'mean',
     'volume':'sum'})
)


quakes.head()


pd.crosstab(index=quakes.magType, columns=quakes.tsunami, values=quakes.mag, aggfunc=max )


quakes.query('magType=="mwb" and tsunami==0')['mag'].max()


( faang
 .groupby('ticker')
 .rolling(60)
 .agg({
     'open':'mean',
     'high':'max',
     'low':'min',
     'close':'mean',
     'volume':'sum'})
).head(100)


faang.head()


pd.pivot_table(faang, index='ticker', aggfunc='mean', values=['open', 'high', 'low', 'close', 'volume'])


(
    faang
    .query('ticker=="AMZN"')
    .loc['2018-Q4']
    .drop(columns='ticker')
    .apply(
        lambda x: x.sub(x.mean()).div(x.std())
    )
)
    


faang.head(3)


event_df = pd.DataFrame({
    'ticker': 'FB',
    'date':  ['2018-07-25', '2018-03-19', '2018-03-20'],
    'event': ['Disappointing user growth announced after close.', 'Cambridge Analytica story', 'FTC investigation']
    })

event_df.head()


event_df.info() # first time through, I hadn't realized `date` wasn't a datetime in this dataframe


# make date datetime
event_df['date'] = pd.to_datetime(event_df['date'])
event_df.info()





# put date and ticker into the index
event_df = event_df.set_index(['date', 'ticker'])


event_df


faang.info() # faang's index IS datetime








# date was already in the index, reset_index kicks the existing index into the values
# to ADD ticker to the index, need to specify append=True

faang.set_index('ticker', append=True).sample(n=5)


faang_w_events= (
    faang
    .set_index('ticker', append=True) # add ticker to date in the index
    .merge(
        event_df, #needs to have date and ticker in the index...see above
        how='outer',
        left_index=True, # to merge on an index as opposed to a dataframe column
        right_index=True # ditto
    )
)

faang_w_events.head()


event_df.index


faang_w_events.loc[[x[0]  for x in event_df.index], :] #just picks out the dates to confirm they're in the merged dataframe


faang.shape


(
    faang
    .groupby('ticker')
    .transform('first')
    .loc[lambda x: x.index <= '2018-01-04']
    
)


faang.drop(columns='ticker').div(faang.groupby('ticker').transform('first'))


faang.set_index('ticker', append=True).sample(10)


(
    faang
    .set_index('ticker', append=True)
    .apply(
        lambda x: x.groupby(['ticker']).transform('first')
    )
)
        
        


faang_indexed = \
(faang
 .set_index('ticker', append=True) # put both `date` and `ticker` into faang's index
 .div(faang                                  # divide by the first value of each ticker, using what we did in the cell above
      .set_index('ticker', append=True)
      .apply(lambda x: x.groupby(['ticker']).transform('first'))
     )
)

faang_indexed.head()



faang_indexed.query('date <= "2018-01-04"')


faang_indexed.query('ticker == "GOOG"')



faang_indexed.query('date <= "2018-01-03"').loc[:, ['high', 'low']]


idx = pd.IndexSlice


idx


faang_indexed.loc[idx[:, "GOOG"],]





faang_indexed.loc[idx[:, ["GOOG", "NFLX"]],]


faang_indexed.loc[idx[:, ["AMZN", "FB"]], ['low', 'volume']]


get_ipython().getoutput("head -5 ~/Downloads/data.csv")


covid = pd.read_csv('~/Downloads/data.csv', index_col='dateRep', parse_dates=True, dayfirst=True)
covid.head()


covid.info()


#?pd.read_csv


#??pd.DataFrame.replace


covid.shape


covid = covid.query('dateRep <= "2020-09-18"')
covid.shape


covid.query('countriesAndTerritories in ["United_States_of_America", "United_Kingdom"]').sample(8)


covid_u = covid.replace({'United_States_of_America':'USA', 'United_Kingdom':'UK'})


covid_u.query('countriesAndTerritories in ["United_States_of_America", "United_Kingdom"]')


covid_u.sort_index(inplace=True)


covid_u.head()


covid_u.tail()


covid_u.index.max()


covid_5=\
(covid
 .assign(
     
     tot_cases = lambda x: x.groupby('countriesAndTerritories').cases.transform('sum'),
     case_rank = lambda x: x.tot_cases.rank(method='dense', ascending=False)
     
 ).query('case_rank <= 5')
)


covid_5.groupby('countriesAndTerritories').cases.max()


covid_5.groupby('countriesAndTerritories').cases.idxmax()


covid_5.groupby('countriesAndTerritories').cases.idxmax()


covid.groupby('countriesAndTerritories').cases.sum().nlargest(5)






