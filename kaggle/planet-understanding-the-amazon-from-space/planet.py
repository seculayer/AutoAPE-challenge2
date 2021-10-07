%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *

import fastai.utils.collect_env; 

fastai.utils.collect_env.show_install(1)
path = Path('/kaggle/input/')
path.ls()

!ls /kaggle/input

!ls /kaggle/input/train-jpg -l |grep "^-"|wc -l
!ls /kaggle/input/test-jpg-v2 -l |grep "^-"|wc -l

df = pd.read_csv(path/'train_v2.csv')
df.head()

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0., max_rotate=15.)

np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')#.use_partial_data(0.01)
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))

data.show_batch(rows=3, figsize=(12,9))

arch = models.densenet121

acc_02 = partial(accuracy_thresh, thresh=0.19)
f_score = partial(fbeta, thresh=0.19)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score], model_dir='/kaggle/working/')

learn.lr_find()
learn.recorder.plot()

lr = 0.01
learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-rn50')

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr/5))

learn.save('stage-2-rn50')

data = (src.transform(tfms, size=256)
        .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data

data.train_ds[0][0].shape

learn.freeze()

learn.lr_find()

learn.recorder.plot()

lr=1e-2/2
learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-256-rn50')
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()

learn.save('stage-2-256-rn50')
learn.export(fname='/kaggle/working/export.pkl',destroy=True)

test = (ImageList.from_folder(path/'test-jpg-v2'))#.use_partial_data(0.01))
len(test)
learn_test = load_learner('/kaggle/working/', test=test, num_workers=0, bs=1)
preds, _ = learn_test.get_preds(ds_type=DatasetType.Test)
preds_tta, _ = learn_test.TTA(ds_type=DatasetType.Test)
#preds = np.mean(np.exp(log_preds))

thresh = 0.15
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_015.csv', index=False)

thresh = 0.18
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_018.csv', index=False)

thresh = 0.19
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_019.csv', index=False)

thresh = 0.20
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_020.csv', index=False)

thresh = 0.21
labelled_preds = [' '.join([learn_test.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
fnames = [f.name[:-4] for f in learn_test.data.test_ds.x.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv('submission_021.csv', index=False)