import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2

%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output
print(check_output(["ls", "../input/intel-mobileodt-cervical-cancer-screening/train/train"]).decode("utf8"))

from glob import glob
basepath = '../input/intel-mobileodt-cervical-cancer-screening/train/train'

all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    print(path)
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    print(len(cervix_images))
    all_cervix_images = all_cervix_images + cervix_images

for n in range(1,4):
    path = '../input/intel-mobileodt-cervical-cancer-screening/additional_Type_' + str(n) + '_v2/Type_' + str(n)
    print(path)
    cervix_images = sorted(glob(path+"/*"))
    print(len(cervix_images))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath':all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row:row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row:row.imagepath.split("/")[-2], axis=1)

type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')
type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis = 1)

fig, axes = plt.subplot(nrows=2, ncols=1, figsize=(10, 8))

type_aggregation.plot.barh(ax=axes[0])
axes[0].set_xlabel("image count")
type_aggregation_p.plot.barh(ax=axes[1])
axes[1].set_xlabel("training size fraction")

all_cervix_images['type'].value_counts()

len(all_cervix_images['type'])

sub = pd.read_csv('../input/intel-mobileodt-cervical-cancer-screening/sample_submission_stg2.csv')
sub.to_csv('sub.csv', index=False)