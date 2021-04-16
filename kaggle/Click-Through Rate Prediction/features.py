from pyarrow import csv
import preprocess

# 피쳐별로 전처리
# 1. id + cid ( C1,C14,C15,C16,C17,C18,C19,C20,C21 )  + click
# 2. id + click +  banner_pos
# 3. id + click + app_id + app_domain + app_category
# 4. id + click + site_id + site_domain + site_category
# 5. id + click + device_id + device_ip +  device_type + device_conn_type
# 6. id + click + day + date

# 피쳐 그룹별로 data frame 분할 => 분할된 애들로 학습
def feature_split():
    train_df = csv.read_csv('./new_train_normal.csv').to_pandas()

    print('train dataset feature split')

    cid_features = ['id', 'click', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    banner_features = ['id', 'click', 'banner_pos']
    app_features = ['id', 'click', 'app_id', 'app_domain', 'app_category']
    site_features = ['id', 'click', 'site_id', 'site_domain', 'site_category']
    device_features = ['id', 'click', 'device_id', 'device_ip', 'device_type', 'device_conn_type']
    day_features = ['id', 'click', 'day', 'date']

    preprocess.feature_split(train_df, cid_features)
    preprocess.feature_split(train_df, banner_features)
    preprocess.feature_split(train_df, app_features)
    preprocess.feature_split(train_df, site_features)
    preprocess.feature_split(train_df, device_features)
    preprocess.feature_split(train_df, day_features)

    print("train feature split compeleted.")



# 피쳐 백 ( 히스토리 ) 생성
def train_feature_bag():

    banner_features = ['id', 'click', 'banner_pos']

    print("feature_banner_pos")
    bag_data = csv.read_csv('./features/feature_banner_pos.csv').to_pandas()
    for i in range(len(banner_features)) :
        print(f'train set processing ... {i} / {len(banner_features)}')
        data = preprocess.bag_features(bag_data, banner_features[i])
    bag_data = data
    bag_data.to_csv('./train_bag/train_banner_bag_features.csv')

    banner_features = ['id', 'banner_pos']

    bag_data = csv.read_csv('./test_features/feature_banner_pos.csv').to_pandas()
    for i in range(len(banner_features)) :
        print(f'test set processing ... {i} / {len(banner_features)}')
        data = preprocess.bag_features(bag_data, banner_features[i])
    bag_data = data
    bag_data.to_csv('./test_bag/test_banner_bag_features.csv')


    print('test feature bag generated.')



# 클릭 히스토리 컬럼 생성
def click_history(bag_data):
    click_history = csv.read_csv('./features/feature_C1').to_pandas()
    his_data = preprocess.click_history(bag_data)
    his_data.to_csv('./id_click_history.csv')
    print('click history generated.')


