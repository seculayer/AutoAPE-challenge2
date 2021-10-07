import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import warnings

train_images = glob("./train_img/*jpg")                                 # train_img 리스트 불러와서 저장
test_images = glob("./test_img/*jpg")                                   # test_img 리스트 불러와서 저장
train_data = np.loadtxt("./train.csv", dtype=np.str, delimiter = ",")   # train.csv 넘파이 텍스트 형태로 불러옴
train_data = train_data[1:,:]                                           # 불러온 train.csv 헤더 떼고 저장
train_data = train_data.tolist()                                        # 헤더 뗀 train.csv를 리스트 형태로 변환

train_data_2 = []           # 사용할 각 리스트 생성
train_data_img = []
train_data_id = []

for i in range(len(train_data)):            # 리스트를 불러와서 위에서부터 순서대로 가져오면서 for문 실행
    train_data_2.append(list(map( lambda x : "./train_img/"+x, train_data[i][0:])))         # train_data의 img행과 train_img 경로를 매핑하여 리스트로 만듬
    train_data_img.append(train_data_2[i][0])                                               # train_data_2의 방금 만들어진 놈 train_data_img에 저장
    train_data_id.append(train_data[i][1])                                                  # train_data의 1열의 id를 train_data_id에 저장
MappingDict = dict(zip(train_data_img, train_data_id))              # 순서대로 매핑된 해당 이미지의 경로와 이미지 아이디 결과를 딕셔너리 형태로 만듬

def TransImage(imgfile):                                            # 이미지 파일을 변환
    img = Image.open(imgfile).convert('LA').resize((64,64))         # 이미지를 받아와서 흑백 변환하고 사이즈 64, 64로 변환
    return np.array(img)[:,:,0]                                     # 변환된 이미지 리턴

train_img = np.array([TransImage(imgfile) for imgfile in train_images])     # 리턴된 변환 이미지를 넘파이 배열 형태로 train_img에 저장
train_id = list(map(MappingDict.get, train_images))                 # MappingDict의 key와 train_images 경로를 매핑하여 일치하는 id를 추출. train_id에 저장

l_encoder = LabelEncoder()
Labeled_id = l_encoder.fit_transform(train_id)                      # 텍스트 형태인 train_id를 레이블 인코딩
oh_encoder = OneHotEncoder()
Onehot_id = oh_encoder.fit_transform(Labeled_id.reshape(-1,1))      # 레이블 인코딩된 train_id를 원핫인코딩(2차원으로 변환하여 적용함)

train_img = train_img.reshape((-1,64,64,1))             # train_img 차원 축소해서 64, 64, 1 형태로 만든다(흑백)
input_shape = train_img[0].shape                        # 학습을 위한 input_shape
x_train = train_img.astype("float32")                   # train_img 정수형 변환해서 x_train으로 저장
y_train = Onehot_id.toarray()                           # 원핫인코딩된 이상한 형태의 id를 배열형태로 변환하여 y_train으로 저장

# cnn
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides = (1,1), input_shape = input_shape))        # 위의 input_shape(64, 64)사이즈 흑백(1)
model.add(tf.keras.layers.BatchNormalization(axis = 3))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides = (1,1)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides = (1,1)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dropout(0.8))
model.add(tf.keras.layers.Dense(4251, activation='softmax'))                                                # y_train 사이즈 그대로 출력 뽑아냄
model.summary()

# 학습
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=150, batch_size=100, verbose=1)

plt.plot(history.history['accuracy'])
plt.title('CNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()

with open("sample_submission.csv","w") as f:                                        # sample_submission.csv 추출
    with warnings.catch_warnings():
        f.write("Image,Id\n")                                                       # 생성파일 헤더
        warnings.filterwarnings("ignore",category=DeprecationWarning)               # 경고무시
        for image in test_images:                                                   # 테스트 이미지 경로에서 하나씩 경로 가져온다.
            img = TransImage(image)                                                 # train 이미지와 마찬가지로 테스트 이미지 형태 맞춰줌(넘파이 배열)
            test_x = img.astype("float32")                                          # 이미지를 정수형으로 변환 후 text_x로 저장
            test_result = model.predict_proba(test_x.reshape(1,64,64,1))            # text_x를 2차원의 64, 64 크기의 흑백으로 변환하여 검증(=test_result)
            predicted_arg = np.argsort(test_result)[0][::-1][:5]                    # 정렬된 test_result 인덱스 반환하고 내림차순 정렬하여 5번째거까지만 저장
            predicted_ori = l_encoder.inverse_transform(predicted_arg)              # 정수 레이블을 문자열로 변환
            image = image.rsplit('/', 1)[1]                                         # test_images 중 현재 이미지의 경로를 스플릿해서 이미지 파일명만 가져옴
            predicted_ori = " ".join(predicted_ori)                                 # 배열 형으로 된거 공백으로 구분하여 다 붙임
            f.write("%s,%s\n" %(image, predicted_ori))                              # 헤더 아래로 쭉 덮어 쓴다
