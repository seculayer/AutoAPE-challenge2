import pandas as pd

fnames = glob('../input/quickdraw-doodle-recognition/train_simplified/*.csv')

cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
drawlist = []

for f in fnames[0:6]:
    first = pd.read_csv(f, nrows=10)
    first = first[first.recognized==True].head(2)
    drawlist.append(first)
draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames)

evens = range(0,11,2)
odds = range(1,12,2)
df1 = draw_df[draw_df.index.isin(evens)]
df2 = draw_df[draw_df.index.isin(odds)]

example1s = [ast.literal_eval(pts) for pts in df1.drawing.values]
example2s = [ast.literal_eval(pts) for pts in df2.drawing.values]

labels = df2.word.tolist()
for i, example in enumerate(example1s):
    plt.figure(figsize=(6,3))

    for x,y in example:
        plt.subplot(1,2,1)
        plt.plot(x,y, marker=".")
        plt.axis('off')

    for x,y in example2s[i]:
        plt.subplot(1,2,2)
        plt.plot(x,y, marker=".")
        plt.axis('off')
        label = labels[i]
        plt.title(label, fontsize=10)
    plt.show()


classfiles = os.listdir('../input/quickdraw-doodle-recognition/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)}

num_classes = 340
imheight, imwidth = 32, 32
ims_per_class = 2000


def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],stroke[1][i],stroke[0][i+1],stroke[1][i]],fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255

train_grand = []
class_paths = glob('../input/quickdraw-doodle-recognition/train_simplified/*.csv')
for i, c in enumerate(tqdm(class_paths[0:num_classes])):
    train = pd.read_csv(c, usecols=['drawing','recognized'], nrows=ims_per_class*5//4)
    train = train[train.recognized == True].head(ims_per_class)
    imagebag = bag.from_sequence(train.drawing.values).map(draw_it)
    trainarray = np.array(imagebag.compute())
    trainarray = np.reshape(trainarray, (ims_per_class, -1))
    labelarray = np.full((train.shape[0], 1), i)
    trainarray = np.concatenate((labelarray, trainarray), axis=1)
    train_grand.append(trainarray)

train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)])
train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

del trainarray
del train

valfrac = 0.1
cutpt = int(valfrac * train_grand.shape[0])
np.random.shuffle(train_grand)
y_train, X_train = train_grand[cutpt:, 0], train_grand[cutpt:, 1:]
y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:]

del train_grand

y_train = keras.utils.to_categorical(y_train, num_classes)
X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
y_val = keras.utils.to_categorical(y_val, num_classes)
X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(680, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

def top_3_accuracy(x,y):
    t3 = top_k_categorical_accuracy(x,y,3)
    return t3

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)
earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience =5)
callbacks = [reduceLROnPlat, earlystop]

model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 22,
          validation_data = (X_val, y_val),
          callbacks=callbacks,
          verbose =1)

ttvlist = []
reader = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', index_col=['key_id'], chunksize=2048)

for chunk in tqdm(reader, total=55):
    imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)
    testarray = np.array(imagebag.compute())
    testarray = np.reshape(testarray, (testarray.shape[0],imheight, imwidth, 1))
    testpreds = model.predict(testarray, verbose=0)
    ttvs = np.argsort(-testpreds)[:, 0:3]
    ttvlist.append(ttvs)

ttvarray = np.concatenate(ttvlist)

preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second':ttvarray[:,1], 'third': ttvarray[:,2]})
preds_df=preds_df.replace(numstonames)
preds_df['words'] = preds_df['first']+" "+preds_df['second']+" "+preds_df['third']

sub = pd.read_csv('../input/quickdraw-doodle-recognition/sample_submission.csv', index_col=['key_id'])
sub['word']=preds_df.words.values
sub.to_csv('subcnn_small.csv')

