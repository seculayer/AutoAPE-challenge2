import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 클릭 수 통계 : 클릭안함 33563901 , 클릭함 6865066
def show_clicks(data):
    data_cnt = data['click'].value_counts()

    plt.title("Clicks", fontsize=15)
    plt.xticks(np.arange(0, 2), labels=['not click', 'click'])

    data_df = pd.DataFrame(data_cnt)
    print(data_df)

    plt.bar(np.arange(len(data_df['click'])), data_cnt, color='green', alpha=.3)
    # plt.grid()
    plt.show()



# 전체 Pie chart - cid, site, app, device
def show_pie(data, col_vals, title):

    for col in col_vals :

        cid_cnt = data[col].value_counts()
        plt.title(title+ " : " + col , fontsize=15)

        cid_labels = data[col].drop_duplicates()

        plt.pie(cid_cnt, labels=cid_labels, autopct='%.1f%%')
        plt.show()



#  시간당 클릭 수 통계
def show_hours_click (data):

    click_data = data[ data['click']==1]

    day_dict = {
        0 : 'MON',
        1 : 'TUE',
        2 : 'WED',
        3 : 'THUR',
        4 : 'FRI',
        5 : 'SAT',
        6 : 'SUN'
    }

    # 1) 요일 x 클릭했다 : 클릭한 애들 중 / 요일 바 통계
    plt.title("Clicks number x days", fontsize=15)
    day_cnt_df = pd.DataFrame(click_data['day'].value_counts())
    day_cnt_df = day_cnt_df.reset_index()

    day_cnt = click_data['day'].value_counts()

    labels = []
    for idx in range(len(day_cnt_df)):
        labels.append(day_dict.get(day_cnt_df.loc[idx,'index']))

    plt.xticks(np.arange(len(labels)), labels=labels)
    plt.bar(np.arange(len(day_cnt)), day_cnt, color='coral', alpha=.7)
    plt.show()



    # 2) 시간 x 클릭했다 : 클릭한 애들 중 / 시간 통계
    plt.title("Clicks x Hours", fontsize=15)
    hour_cnt = click_data['hour'].value_counts()

    plt.xticks(np.arange(len(hour_cnt)), labels=hour_cnt.index)
    plt.bar(np.arange(len(hour_cnt)), hour_cnt, color='teal', alpha=.7)
    plt.show()


    # 3) 요일 x 시간 x 클릭 :  x : 클릭값 , y : 시간, label : 요일
    all_data = data[['hour', 'day', 'click']]
    grp_data = pd.DataFrame(all_data.groupby(['hour', 'day']).count())
    grp_data.index.name = 'id'
    grp_df = grp_data.reset_index()

    # 요일별 분류, x : 클릭값 , y : 시간
    markers = ['o', 'x', '^', 'h', '*', 'p', 's']
    colors=['red', 'blue', 'green', 'orange', 'purple', 'pink', 'coral']

    label_chk = []
    for idx in range(len(grp_df)):
        x = grp_df.loc[idx, 'click']
        y = grp_df.loc[idx, 'hour']
        today = int(grp_df.loc[idx, 'day'])

        day_str = day_dict.get(today)

        if day_str in label_chk :
            plt.scatter(x, y, marker=markers[today], color=colors[today], label='_nolegend_')
        else :
            label_chk.append(day_str)
            plt.scatter(x, y, marker=markers[today], color=colors[today], label=day_str)

    plt.legend()
    plt.show()



    #  4) 날짜별 클릭 통계
    click_data = data[data['click']==1]

    plt.title("Clicks number x dates", fontsize=15)
    date_cnt_df = pd.DataFrame(click_data['date'].value_counts())
    date_cnt_df = date_cnt_df.reset_index()

    date_cnt = click_data['date'].value_counts()

    labels = []
    for idx in range(len(date_cnt_df)):
        labels.append(date_cnt_df.loc[idx,'index'])

    plt.xticks(np.arange(len(labels)), labels=labels)
    plt.bar(np.arange(len(date_cnt)), date_cnt, color='lightslategray', alpha=.7)
    plt.show()






# 사이트 아이디, 사이트 도메인, 사이트 카테고리,
# 앱 아이디, 도메인, 카테고리,
# 디바이스 아이디, 아이피, 모델, 타입 각 종류마다 개수 계산

#  app id, domain : 종류 너무 여러개
# stite domain : 종류 너무 여러개 임
def show_how_many(data):
    # 사이트 정보 계산


    site_label = ['site_id', 'site_domain', 'site_category']
    site_info = [len(data['site_id'].drop_duplicates()),
                 len(data['site_domain'].drop_duplicates()),
                 len(data['site_category'].drop_duplicates())
                 ]

    print("site_category")
    print(data['site_category'].drop_duplicates())

    plt.title("Number of site id / domain / category classes")
    plt.xticks(np.arange(len(site_label)), labels=site_label)
    plt.bar(np.arange(len(site_info)), site_info, color='seagreen', alpha=.7)
    plt.show()


    app_label = ['app_id', 'app_domain', 'app_category']
    app_info = [len(data['app_id'].drop_duplicates()),
                 len(data['app_domain'].drop_duplicates()),
                 len(data['app_category'].drop_duplicates())
                 ]

    print("app_category")
    print(data['app_category'].drop_duplicates())

    plt.title("Number of app id / domain / category classes")
    plt.xticks(np.arange(len(app_label)), labels=app_label)
    plt.bar(np.arange(len(app_info)), app_info, color='blue', alpha=.1)
    plt.show()


    day_label = ['day', 'date']
    day_info = [len(data['day'].drop_duplicates()),
                len(data['date'].drop_duplicates()),
                ]
    print("day")
    print(data['day'].drop_duplicates())
    print(data['date'].drop_duplicates())

    plt.title("Number of day / date ")
    plt.xticks(np.arange(len(day_label)), labels=day_label)
    plt.bar(np.arange(len(day_info)), day_info, color='blue', alpha=.1)
    plt.show()

    # dev_label = ['device_id', 'device_model', 'device_type']
    # dev_info = [len(data['device_id'].drop_duplicates()),
    #              len(data['device_model'].drop_duplicates()),
    #              len(data['device_type'].drop_duplicates()),
    #              ]

    dev_label = ['device_type', 'device_conn_type']
    dev_info = [len(data['device_conn_type'].drop_duplicates()),
                len(data['device_type'].drop_duplicates())]

    print("device_type")
    print(data['device_type'].drop_duplicates())

    print("device_conn_type")
    print(data['device_conn_type'].drop_duplicates())


    plt.title("Number of device id / ip / model / type / conn type classes")
    plt.xticks(np.arange(len(dev_label)), labels=dev_label)
    plt.bar(np.arange(len(dev_info)), dev_info, color='orange', alpha=.7)
    plt.show()

    print('site, app, devcie 총 통계')
    print(f'site id: {site_info[0]}, domain: {site_info[1]}, category: {site_info[2]}')
    print(f'app id: {app_info[0]}, domain: {app_info[1]}, category: {app_info[2]}')
    print(f'day : {day_info[0]}, date: {day_info[1]}')
    print(f'device type: {dev_info[0]}, conn type : {dev_info[1]}')



# ['site_id', 'site_category', 'site_domain']
# ['app_id', 'app_category', 'app_domain']
# ['device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']

# 사이트, 앱, 디바이스별 클릭 분석
def show_clicked_info(data, kind):
    click_data = data[data['click']==1]
    for idx in range(len(kind)):
        title = "Clicks number x " + kind[idx]

        plt.title(title, fontsize=15)
        id_cnt_df = pd.DataFrame(click_data[kind[idx]].value_counts())
        id_cnt_df = id_cnt_df.reset_index()

        id_cnt = click_data[kind[idx]].value_counts()

        labels = []
        if len(id_cnt) > 10 :
            labels = [ i for i in range(len(id_cnt))]
        else :
            for k in range(len(id_cnt_df)):
                labels.append(id_cnt_df.loc[k,'index'])

        plt.xticks(np.arange(len(labels)), labels=labels, fontsize=6)
        plt.bar(np.arange(len(id_cnt)), id_cnt, color='lightslategray', alpha=.7)
        plt.show()

        print(kind)
        print(idx)
        print(f'클릭된 - {kind[idx]} 총 개수  : {len(id_cnt)}')
        print("클릭된 - " + kind[idx] + " top7 총 통계 ")
        print(id_cnt_df[:7])



# 앱 사이트 안에 몇개의 도메인, 카테고리가 있는지?
# df, kinds = ['app_id' 'app_domain', 'app_category']
def show_num_kinds(data, kinds):

    #  id x domain x category x banner pos
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[1], kinds[2], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(grp_data.head())

    print(f'id x domain x category x banner pos : {len(grp_data)}')



    #  banner_pos x category
    grp_data = pd.DataFrame(data.groupby([kinds[2], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())

    print(f' banner_pos x category : {len(grp_data)}')



    #  id x domain x category 96432
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[1], kinds[2]]).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())

    print(f'id x domain x category pos : {len(grp_data)}')



    #  domain x category  1730
    grp_data = pd.DataFrame(data.groupby([kinds[1], kinds[2]]).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'domain x category pos : {len(grp_data)}')


    #  domain x category x banner_pos  1730
    grp_data = pd.DataFrame(data.groupby([kinds[1], kinds[2], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'domain x category x banner_pos : {len(grp_data)}')



    #  id x domain
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[1]]).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'id x domain : {len(grp_data)}')



    #  id x domain x banner_pos 29681
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[1], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'id x domain x banner_pos : {len(grp_data)}')


    #  id x category 16391
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[2]]).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'id x category : {len(grp_data)}')


    #  id x category x banner_pos
    grp_data = pd.DataFrame(data.groupby([kinds[0], kinds[2], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'id x category x banner_pos : {len(grp_data)}')



    #  id x banner_pos
    grp_data = pd.DataFrame(data.groupby([kinds[0], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'id x banner_pos : {len(grp_data)}')



    #  domain x banner_pos
    grp_data = pd.DataFrame(data.groupby([kinds[1], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'domain x banner_pos : {len(grp_data)}')


    #  category x banner_pos
    grp_data = pd.DataFrame(data.groupby([kinds[2], 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'category x banner_pos : {len(grp_data)}')





    #  site_id x site_domain x site_category
    grp_data = pd.DataFrame(data.groupby(['site_id', 'site_domain', 'site_category']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_id x site_domain x site_category : {len(grp_data)}')


    #  site_id x site_domain
    grp_data = pd.DataFrame(data.groupby(['site_id', 'site_domain']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_id x site_domain : {len(grp_data)}')




    #  site_domain  x site_category
    grp_data = pd.DataFrame(data.groupby(['site_domain', 'site_category']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_domain  x site_category : {len(grp_data)}')



    #  site_id  x site_category
    grp_data = pd.DataFrame(data.groupby(['site_id', 'site_category']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_id  x site_category : {len(grp_data)}')


    #  site_id  x banner_pos
    grp_data = pd.DataFrame(data.groupby(['site_id', 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_id  x banner_pos : {len(grp_data)}')



    #  site_domain  x banner_pos
    grp_data = pd.DataFrame(data.groupby(['site_domain', 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_domain  x banner_pos : {len(grp_data)}')



    #  site_category  x banner_pos
    grp_data = pd.DataFrame(data.groupby(['site_category', 'banner_pos']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'site_category  x banner_pos : {len(grp_data)}')



    # app id x site_id
    grp_data = pd.DataFrame(data.groupby([kinds[0], 'site_id']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'app id x site_id : {len(grp_data)}')


    #  app_domain  x site_domain
    grp_data = pd.DataFrame(data.groupby(['app_domain', 'site_domain']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'app_domain  x site_domain : {len(grp_data)}')


    #  app_category  x site_category
    grp_data = pd.DataFrame(data.groupby(['app_domain', 'site_domain']).count())
    grp_data = grp_data.reset_index()

    print(grp_data.head())
    print(f'app_category  x site_category : {len(grp_data)}')



# 클릭된 - site_id 총 개수  : 1397 / 1428
# 클릭된 - site_domain 총 개수  : 1445 / 1489
# 클릭된 - app_id 총 개수  : 1070 / 1080
# 클릭된 - app_domain 총 개수  : 95 / 100
def show_fasten_click(data):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    click_data = data[data['click']==1]
    not_click_data = data[data['click']==0]

    grp_data = pd.DataFrame(not_click_data.groupby(['app_category', 'app_domain']).count())
    grp_data = grp_data.reset_index()
    print(grp_data.head())
    print(f'app_category x app_domain : {len(grp_data)}')



# site id, site domain, app_id, app domain 조합 클릭 된애들 / 안된애들
# 조합중에, 클릭이 단 한번도 되지 않은 애들이 있는지 체크하고 드롭
def show_duplicated_clicked(data):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # 클릭된 애들이랑 안된애들 데이터 프레임 나누기
    click_df = data[data['click']==1]
    click_df= click_df.drop(['click'], axis=1)
    click_df = click_df.reset_index()

    not_click_df = data[data['click']==0]
    not_click_df = not_click_df.drop(['click'], axis=1)
    not_click_df = not_click_df.reset_index()

    # 클릭한 아이디 피쳐값들과 중복되는 애들만 클릭안한 df 에서 가져온다
    keys = ['banner_pos',
            'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category',
            'device_type', 'date', 'day']

    same_idx = []
    for i in range(len(click_df)):
        print(f'{i} / {len(click_df)}...........')
        dict = {'banner_pos':[],
                'site_id':[], 'site_domain':[], 'site_category':[],
                'app_id':[], 'app_domain':[], 'app_category':[],
                'device_type':[], 'date':[], 'day':[]}

        for k in range(len(keys)):
            dict[keys[k]] = [click_df.loc[i, keys[k]]]

        chk_df = pd.DataFrame(not_click_df.isin(dict).all(axis=1))
        idx = chk_df[0] == True
        if len(idx) > 0 :
            print(click_df[idx])
            idx = idx.tolist()
            for j in idx:
                same_idx.append(j)

    print(same_idx)
