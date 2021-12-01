import pandas as pd

train=pd.read_csv('/kaggle/input/facebook-recruiting-iv-human-or-bot/train.csv.zip')
train.head()

test = pd.read_csv('/kaggle/input/facebook-recruiting-iv-human-or-bot/test.csv.zip')
test.head()

bid = pd.read_csv('/kaggle/input/facebook-recruiting-iv-human-or-bot/bids.csv.zip')
bid.head()

bid.country.unique()

sub = pd.read_csv('/kaggle/input/facebook-recruiting-iv-human-or-bot/bids.csv.zip')
sub.head()

robot = train('outcome') == 1.0

val = 103/2013

for c in range(len(sub)):
    sub.iloc[c, sub.columns.get_loc('prediction')]=val

sub.to_csv('sub.csv', index=False)
