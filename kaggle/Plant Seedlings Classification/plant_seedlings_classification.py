import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.model_selection import train_test_split
import xgboost as xgb

p = '../249_plant/data'

def df_of_images(folder_name, path='../249_plant/data/'):
    itms = list()
    for x in os.listdir(os.path.join(path, folder_name)):
        for img in os.listdir(os.path.join(path, folder_name, x)):
            itms.append({
                'label': x.lower().strip().replace(' ', '_').replace('-', '_'),
                'image_path': os.path.join(path, folder_name, x, img)
            })
    return pd.DataFrame(itms)

def extract_features_keras(image_path,model):
   img = image.load_img(image_path, target_size=(299, 299))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   predictions = model.predict(x)
   return np.squeeze(predictions)



if __name__ == '__main__':

    train = df_of_images('train')
    test = pd.DataFrame({'image_path': [os.path.join(p, 'test', i) for i in
                                        os.listdir('../249_plant/data/test/')]})
    base_model = InceptionV3()
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    train['image_features'] = train.image_path.apply(lambda x: extract_features_keras(x, base_model))
    test['image_features'] = test.image_path.apply(lambda x: extract_features_keras(x, base_model))

    train_, test_ = train_test_split(train, test_size=0.33, random_state=42, stratify=train.label)

    'train:', train_.label.value_counts() / len(train_), 'test:', test_.label.value_counts() / len(test_)

    xgc = xgb.XGBClassifier(objective='multi:softmax', num_class=train.label.nunique())
    xgc.fit(pd.DataFrame(train_['image_features'].values.tolist()), train_.label)

    results = test_.copy()
    results['y_pred'] = xgc.predict(pd.DataFrame(test_['image_features'].values.tolist()))

    label_map = {x.lower().strip().replace(' ', '_').replace('-', '_'): x for x in os.listdir(os.path.join(p, 'train'))}

    results = test.copy()
    results['species'] = xgc.predict(pd.DataFrame(test['image_features'].values.tolist()))
    results['species'] = results['species'].replace(label_map)
    results['file'] = results.image_path.apply(lambda x: x.split('/')[-1])

    results[['file', 'species']].to_csv('submission.csv', index=False)

