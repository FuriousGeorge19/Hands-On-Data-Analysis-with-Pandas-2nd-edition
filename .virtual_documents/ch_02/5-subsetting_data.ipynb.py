import pandas as pd

df = pd.read_csv('data/earthquakes.csv')


df.mag


df['mag']


df[['mag', 'title']]


df[ for c in df.columns if c.str.]




















df[
    ['title', 'time']
    + [col for col in df.columns if col.startswith('mag')]
]


[col for col in df.columns if col.startswith('mag')]


['title', 'time'] \
+ [col for col in df.columns if col.startswith('mag')]


df[
    ['title', 'time']
    + [col for col in df.columns if col.startswith('mag')]
]


df[100:103]


df[['title', 'time']][100:103]


df[100:103][['title', 'time']].equals(
    df[['title', 'time']][100:103]
)


df[110:113]['title'] = df[110:113]['title'].str.lower()


df[110:113]['title']


df.loc[110:112, 'title'] = df.loc[110:112, 'title'].str.lower()
df.loc[110:112, 'title']


df.loc[:,'title']


df.loc[10:15, ['title', 'mag']]


df.iloc[10:15, [19, 8]]


df.iloc[10:15, 6:10]


df.iloc[10:15, 6:10].equals(
    df.loc[10:14, 'gap':'magType']
)


df.at[10, 'mag']


df.iat[10, 8]


df.mag > 2


df[df.mag >= 7.0]


df.loc[
    df.mag >= 7.0,
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    (df.tsunami == 1) & (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    (df.tsunami == 1) | (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    (df.place.str.contains('Alaska')) & (df.alert.notnull()),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    (df.place.str.contains(r'CA|California$')) & (df.mag > 3.8),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    df.mag.between(6.5, 7.5),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.loc[
    df.magType.isin(['mw', 'mwb']),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


[df.mag.idxmin(), df.mag.idxmax()]


df.loc[
    [df.mag.idxmin(), df.mag.idxmax()],
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
]


df.filter(items=['mag', 'magType']).head()


df.filter(like='mag').head()


df.filter(regex=r'^t').head()


df.set_index('place').filter(like='Japan', axis=0).filter(items=['mag', 'magType', 'title']).head()


df.set_index('place').title.filter(like='Japan').head()
