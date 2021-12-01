import numpy as np
import pandas as pd

import os

d1 = pd.read_csv('/kaggle/input/inclusive-images-challenge/tuning_labels.csv', names=['id', 'labels'], index_col=['id'])
d2 = pd.read_csv('/kaggle/input/inclusive-images-challenge/stage_2_sample_submission.csv', index_col='image_id')
d2['labels'] = ''.join(d1['labels'].str.split().apply(pd.Series).stack().value_counts().head(3).index.tolist())
d2.update(d1)
d2.to_csv('last_day_6_lines_baseline.csv')