train = pd.read_csv('/kaggle/input/talkingdata-mobile-user-demographics/gender_age_train.csv.zip')
train.head()

test = pd.read_csv('/kaggle/input/talkingdata-mobile-user-demographics/gender_age_test.csv.zip')
test.head()

sample = pd.read_csv('/kaggle/input/talkingdata-mobile-user-demographics/sample_submission.csv.zip')
sample.head()

sample.to_csv('sample.csv', index=False)