import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# k-fold 학습 알고리즘
def cross_validaton(train_list, labels, model):
    # Dataset division
    kfold = 5
    div = len(train_list)//kfold

    # train_div = [train_list[i*div:(i+1)*div] for i in range(kfold+1)]
    # train_div = [train_list[i*div:(i+1)*div][:][:][:] for i in range(kfold+1)]

    weights = [ 0 for i in range(22)]

    for del_idx in range(kfold):
        print(f'{del_idx+1}번째 split & training...')

        shape_test = np.asarray(train_list)

        train_fold = train_list[:]
        valid_div = train_fold[del_idx*div:(del_idx+1)*div]
        del train_fold[del_idx*div:(del_idx+1)*div]

        train_fold = np.asarray(train_fold)
        train_fold = train_fold/255

        valid_div = np.asarray(valid_div)
        valid_div = valid_div/255

        label_fold = labels[:]
        valid_label = label_fold[del_idx*div:(del_idx+1)*div]
        del label_fold[del_idx*div:(del_idx+1)*div]

        label_fold = np.asarray(label_fold)
        valid_label = np.asarray(valid_label)

        data_aug_gen = ImageDataGenerator(
            # rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        data_aug_gen.fit(train_fold)


        print(f'train_fold shape , train_fold = {train_fold.shape}')
        print(f'label_fold shape, label_fold = {label_fold.shape}')

        batch_size = 32
        epoch = 50

        chkpoint = ModelCheckpoint(filepath='./', monitor='val_loss', verbose=0, save_weights_only=True, mode='auto')
        early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

        callbacks = [chkpoint, early_stop]

        training = model.fit_generator(
            data_aug_gen.flow(train_fold, label_fold, batch_size=batch_size),
            steps_per_epoch = train_fold.shape[0] // batch_size,
            epochs = epoch,
            validation_data = (valid_div, valid_label),
            callbacks = callbacks
        )

        print(f'이전 weights 길이 : {len(weights)}')
        #  weights += model.get_weights()
        print(f'weigths shape> : {(np.asarray(weights)).shape}')
        temp = model.get_weights()
        weights = [ weights[i]+temp[i] for i in range (len(weights)) ]


    # weight_div = [kfold for i in range(len(weights))]
    # weights = weights/weight_div


    weights = [ weights[i]/kfold for i in range(len(weights)) ]
    model.set_weights(weights)

    return model