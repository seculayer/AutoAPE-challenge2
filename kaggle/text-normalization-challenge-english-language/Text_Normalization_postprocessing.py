# 후처리
new_test_df = pd.read_csv('./make_class_cnn_kfold_7.csv')
n_test_data = new_test_df.values

train_df = pd.read_csv('../input/text-normalization-challenge-english-language/en_train.csv.zip')
train_array = train_df.values

test_df = pd.read_csv('../input/text-normalization-challenge-english-language/en_test_2.csv.zip')
test_df_array = test_df.values
gc.collect()

train_y = train_array[:9000000,2]
l_encoder = LabelEncoder()
train_y = l_encoder.fit_transform(train_y)

test_class = n_test_data[:,0].astype(int)
test_before = n_test_data[:,1].reshape(-1,1)

test_class = l_encoder.inverse_transform((test_class)).reshape(-1,1)
test_array = np.zeros(shape=(len(test_class),), dtype=np.int8).reshape(-1,1)

test_data = np.concatenate((test_class, test_before), axis=1)
test_data = np.concatenate((test_data, test_array), axis=1)


map={0:"", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven", 8:"eight", 9:"nine", 10:"ten",
     11:"eleven", 12:"twelve", 13:"thirteen", 14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen",
     18:"eighteen", 19:"nineteen", 20:"twenty", 30:"thirty", 40:"forty", 50:"fifty", 60:"sixty", 70:"seventy", 80:"eighty", 90:"ninety",}

cardi_n_map = {'I':'one','II':'two','III':'three','IV':'four','V':'five','VI':'six','VII':'seven','VIII':'eight','IX':'nine','X':'ten',
               'XI':'eleven','XII':'twelve','XIII':'thirteen','XIV':'fourteen','XV':'fifteen','XVI':'sixteen','XVII':'seventeen',
               'XVIII':'eighteen','XIX':'nineteen','XX':'twenty'}

point_map = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven", 8:"eight", 9:"nine", 0:"o"}

dict_mon = {"1": "January", "2": "February", "3": "march", "4": "april", "5": "may ","6": "june", "7": "july", "8": "august","9": "september",
            "01": "January", "02": "February", "03": "march", "04": "april", "05": "may ","06": "june", "07": "july", "08": "august","09": "september",
            "10": "october","11": "november","12": "december",
            'jan': "January", "feb": "February", "mar ": "march", "apr": "april", "may": "may ","jun": "june", "jul": "july", "aug": "august","sep": "september",
            "oct": "october","nov": "november","dec": "december", "january":"January", "february":"February", "march":"march","april":"april", "may": "may",
            "june":"june","july":"july", "august":"august", "september":"september", "october":"october", "november":"november", "december":"december"}

dict_day = {'01':'first','1':'first','02':'second','2':'second','03':'third','3':'third','04':'fourth','4':'fourth','05':'fifth','5':'fifth','06':'sixth','6':'sixth','07':'seventh','7':'seventh',
            '08':'eighth','8':'eighth','09':'ninth','9':'ninth','10':'tenth','11':'eleventh','12':'twelves','13':'thirteenth','14':'fourteenth','15':'fifteenth',
            '16':'sixteenth','17':'seventeenth','18':'eighteenth','19':'nineteenth','20':'twentieth','21':'twenty first','22':'twenty second','23':'twenty third','24':'twenty fourth',
            '25':'twenty fifth','26':'twenty sixth','27':'twenty seventh','28':'twenty eighth','29':'twenty ninth','30':'thirtieth','31':'thirty first'}

dict_measure = {'"': 'inches', "'": 'feet', 'km/s': 'kilometers per second', 'AU': 'units', 'BAR': 'bars', 'CM': 'centimeters', 'mm': 'millimeters', 'FT': 'feet', 'G': 'grams',
                'GAL': 'gallons', 'GB': 'gigabytes', 'GHZ': 'gigahertz', 'HA': 'hectares', 'HP': 'horsepower', 'HZ': 'hertz', 'KM':'kilometers', 'km3': 'cubic kilometers',
                'KA':'kilo amperes', 'KB': 'kilobytes', 'KG': 'kilograms', 'KHZ': 'kilohertz', 'KM²': 'square kilometers', 'KT': 'knots', 'KV': 'kilo volts', 'M': 'meters',
                'KM2': 'square kilometers','Kw':'kilowatts', 'KWH': 'kilo watt hours', 'LB': 'pounds', 'LBS': 'pounds', 'MA': 'mega amperes', 'MB': 'megabytes', 'sq ft':'square feet',
                'KW': 'kilowatts', 'MPH': 'miles per hour', 'MS': 'milliseconds', 'MV': 'milli volts', 'kJ':'kilojoules', 'km/h': 'kilometers per hour',  'V': 'volts',
                'M2': 'square meters', 'M3': 'cubic meters', 'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces',  'MHZ': 'megahertz', 'MI': 'miles',
                'MB/S': 'megabytes per second', 'MG': 'milligrams', 'ML': 'milliliters', 'YD': 'yards', 'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams',
                'gal': 'gallons', 'gb': 'gigabytes', 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz', 'kWh': 'kilo watt hours', 'ka': 'kilo amperes', 'kb': 'kilobytes',
                'kg': 'kilograms', 'khz': 'kilohertz', 'km': 'kilometers', 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots','kv': 'kilo volts', 'kw': 'kilowatts',
                'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters','m3': 'cubic meters', 'ma': 'mega amperes', 'mb': 'megabytes', 'mb/s': 'megabytes per second',
                'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles', 'ml': 'milliliters', 'mph': 'miles per hour','ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts', 'm²': 'square meters',
                'm³': 'cubic meters', 'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms', 'ΜG': 'micrograms', 'kg/m3': 'kilograms per meter cube', 'sq':'square', 'sq mi':'square mile'}

dict_ord = {1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth',11:'eleventh',
            12:'twelfth',13:'thirteenth',14:'fourteenth',15:'fifteenth',16:'sixteenth',17:'seventeenth',18:'eighteenth',19:'nineteenth',
            'I':'first','II':'second','III':'third','IV':'fourth','V':'fifth','VI':'sixth','VII':'seventh','VIII':'eighth','IX':'ninth','X':'tenth',
            'XI':'eleventh','XII':'twelfth','XIII':'thirteenth','XIV':'fourteenth','XV':'fifteenth','XVI':'sixteenth','XVII':'seventeenth',
            'XVIII':'eighteenth','XIX':'nineteenth','XX':'twentieth',
            20:'twentieth',30:'thirtieth',40:'fortieth',50:'fiftieth',60:'sixtieth',70:'seventieth',80:'eightieth',90:'ninetieth'}

dict_verb = {"#":"number","&":"and","α":"alpha","Α":"alpha","β":"beta","Β":"beta","γ":"gamma","Γ":"gamma",
             "δ":"delta","Δ":"delta","ε":"epsilon","Ε":"epsilon","Ζ":"zeta","ζ":"zeta","η":"eta","Η":"eta",
             "θ":"theta","Θ":"theta","ι":"iota","Ι":"iota","κ":"kappa","Κ":"kappa","λ":"lambda","Λ":"lambda",
             "Μ":"mu","μ":"mu","ν":"nu","Ν":"nu","Ξ":"xi","ξ":"xi","Ο":"omicron","ο":"omicron","π":"pi","Π":"pi",
             "ρ":"rho","Ρ":"rho","σ":"sigma","Σ":"sigma","ς":"sigma","Φ":"phi","φ":"phi","τ":"tau","Τ":"tau",
             "υ":"upsilon","Υ":"upsilon","Χ":"chi","χ":"chi","Ψ":"psi","ψ":"psi","ω":"omega","Ω":"omega",
             "$":"dollar","€":"euro","~":"tilde","_":"underscore","ₐ":"sil","%":"percent","³":"cubed","-":"-"}

elec_dic = {'0':'o', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}


def VERBATIM(x):
    try:
        text = str(dict_verb.get(x))
        if text == 'None':
            text = ''
            if re.match(r'[A-Za-z]',x):
                for i in range(len(x)):
                    text += x[i].lower()
                    if i == len(x): break
                    text += ' '
            else:
                text = x
            return text
        return text
    except:
        return x

def CARDINAL(x):
    x = str(x).replace(',','').replace(' ','')
    try:
        x = int(x)
        text = ''
        if x == 0: text = 'zero'

        elif 0 < x <= 20:
            if x == 20: text = map.get(x)
            elif x < 20: text = map.get(x)
        elif 20 < x <= 99:
            t=str(x)[0:1] #십의자리
            o=str(x)[1:2] #일의자리
            key=t+"0" #key
            temp1=map.get(int(key))
            temp2=map.get(int(o))
            if int(o) == 0:
                text = str(temp1)
            else:
                text = str(temp1) + ' ' + str(temp2)

        elif 99 < x < 1000:
            h=str(x)[0:1] #백의자리
            t=str(x)[1:3] #뒤의두자리
            if int(t) != 0:
                if int(t) < 20:
                    text = map.get(int(h)) + ' hundred ' + map.get(int(t))
                elif int(t) >= 20:
                    if int(t) % 10 == 0:
                        text = map.get(int(h)) + ' hundred ' + map.get(int(t))
                    else:
                        text = map.get(int(h)) + ' hundred ' + map.get(int(t[0]+'0')) + ' ' + map.get(int(t[1]))
            else:
                text = map.get(int(h)) + ' hundred'

        elif 1000 <= x < 10000:
            th = str(x)[0] # 천의자리
            h=str(x)[1] #백의자리
            t=str(x)[2:4] #뒤의두자리
            temp1=map.get(int(th)) + ' '
            temp2='thousand '
            if int(h) > 0:
                temp3=map.get(int(h)) + ' '
                temp4='hundred '
            else:
                temp3 = ''
                temp4 = ''
            if int(t) > 19:
                tencount = (int(int(t) / 10)) * 10
                onecount = (int(t) % 10)
                temp5=map.get(int(tencount)) + ' '
                temp6=map.get(int(onecount))
            elif int(t) <= 19 and int(t) > 0:
                temp5=map.get(int(t))
                temp6=''
            else:
                temp5, temp6 = '', ''
            text = str(temp1) + str(temp2) + str(temp3) + str(temp4) +str(temp5) + str(temp6)

        elif 10000 <= x < 20000:
            dth = str(x)[0:2]   # 천 대 두자리
            h=str(x)[2:3] #백의자리
            t=str(x)[3:] #뒤의두자리
            temp1 = map.get(int(dth)) + ' '
            temp2='thousand '
            if int(h) > 0:
                temp3=map.get(int(h)) + ' '
                temp4='hundred '
            else:
                temp3 = ''
                temp4 = ''
            if int(t) > 19:
                tencount = (int(int(t) / 10)) * 10
                onecount = (int(t) % 10)
                temp5=map.get(int(tencount)) + ' '
                temp6=map.get(int(onecount))
            else:
                temp5=map.get(int(t))
                temp6=''
            text = str(temp1) + str(temp2) + str(temp3) + str(temp4) +str(temp5) + str(temp6)

        return text

    except:
        return x


def DECIMAL(x):
    try:
        x = str(x).replace(',','')
        # 띄어쓰기가 있을때
        space_split = x.split()
        split_text = ''
        for i in range(len(space_split)): split_text += space_split[i]

        # 시작이 .이 일때
        if re.match(r'\.', x):
            point_split = split_text.split()
            point_split[0] = point_split[0].split('.')[1]
            behind_num = point_split[0]

            behind_text = ''
            for i in range(len(behind_num)):
                ii = behind_num[i]
                if i+1 == len(behind_num):
                    behind_text += point_map.get(int(ii))
                else:
                    behind_text += point_map.get(int(ii)) + ' '
            point_split[0] = 'point ' + str(behind_text)
            text_array = point_split

        # 시작이 마이너스 일때
        elif re.match(r'\-', x):
            minus_replace = split_text.replace('-','minus ')
            minus_split = minus_replace.split()
            minus_split[0] = str(minus_split[0] + ' ')
            # 마이너스가 있고 point가 있을 때
            mpoint_num = minus_split[1]
            mpoint_array = mpoint_num.split('.')
            front_num = CARDINAL(int(mpoint_array[0]))
            # point 뒷자리
            behind_num = mpoint_array[1]
            behind_text = ''
            for i in range(len(behind_num)):
                ii = behind_num[i]
                if i+1 == len(behind_num):
                    behind_text += point_map.get(int(ii))
                else:
                    behind_text += point_map.get(int(ii)) + ' '
            minus_split[1] = str(front_num) + ' point ' + str(behind_text)
            text_array = minus_split

        # 시작이 숫자이고 point가 있을 때
        else:
            plus_split = split_text.split()
            mpoint_num = plus_split[0]
            mpoint_array = mpoint_num.split('.')
            front_num = CARDINAL(int(mpoint_array[0]))
            # point 뒷자리
            behind_num = mpoint_array[1]
            behind_text = ''
            for i in range(len(behind_num)):
                ii = behind_num[i]
                if i+1 == len(behind_num):
                    behind_text += point_map.get(int(ii))
                else:
                    behind_text += point_map.get(int(ii)) + ' '
            plus_split[0] = str(front_num) + ' point ' + str(behind_text)
            text_array = plus_split
        text = ''
        for i in range(len(text_array)):
            text += text_array[i]
        return text

    except:
        return x

def MONEY(key):
    try:
        key = key.replace(',', '')
        if 'm' in key or 'M' in key or 'million' in key:
            key = key.replace('million','').replace('m','').replace('Million','').replace('M','').replace(' ', '')
            v = key.replace('US$','million dollars ').replace('$','million dollars ').replace('€','million euros ').replace('£','million pounds ').replace('USD','million united states dollars ').rsplit(' ', 1)
        elif 'b' in key or 'B' in key or 'billion' in key:
            key = key.replace('billion','').replace('b','').replace('Billion','').replace('B','').replace(' ', '')
            v = key.replace('US$','billion dollars ').replace('$','billion dollars ').replace('€','billion euros ').replace('£','billion pounds ').replace('USD','billion united states dollars ').rsplit(' ', 1)
        else:
            v = key.replace('US$','dollars ').replace('$','dollars ').replace('€','euros ').replace('£','pounds ').replace('USD','united states dollars ').rsplit(' ', 1)


        if v[0] == 'dollars' or v[0] == 'million dollars' or v[0] == 'billion dollars':
            front = CARDINAL(v[1].rsplit('.', 1)[0])
            behind = DECIMAL('.' + v[1].split('.')[1])
            text = front + ' ' + behind + ' ' + v[0]
        elif v[0] == 'euros' or v[0] == 'million euros' or v[0] == 'billion euros':
            front = CARDINAL(v[1].rsplit('.', 1)[0])
            behind = DECIMAL('.' + v[1].split('.')[1])
            text = front + ' ' + behind + ' ' + v[0]
        elif v[0] == 'pounds' or v[0] == 'million pounds' or v[0] == 'billion pounds':
            front = CARDINAL(v[1].rsplit('.', 1)[0])
            behind = DECIMAL('.' + v[1].split('.')[1])
            text = front + ' ' + behind + ' ' + v[0]
        elif v[0] == 'united states dollars' or v[0] == 'million united states dollars' or v[0] == 'billion united states dollars':
            front = CARDINAL(v[1].rsplit('.', 1)[0])
            behind = DECIMAL('.' + v[1].split('.')[1])
            text = front + ' ' + behind + ' ' + v[0]

        return text.lower()
    except:
        return(key)

def MEASURE(x):
    try:
        if x.endswith('%'):
            percentKey = x.split('%')
            if percentKey[0].isdigit():
                text = CARDINAL(percentKey[0]) + ' percent'
            else:
                text = DECIMAL(percentKey[0]).strip() + ' percent'
            return text
        else:
            # 숫자 + / + 단위
            if re.match(r'\S+/\S+', x):
                unittest = x.split('/')
                if unittest[-1] in dict_measure.keys():
                    val = DECIMAL(unittest[0])
                    unit = dict_measure.get(unittest[-1])
                    text = val + ' per ' + unit
                else:
                    val = DECIMAL(unittest[0])
                    unit = unittest[-1]
                    text = val + ' per ' + unit
                return text
            # 숫자+ 공백 + / + 단위
            elif re.match(r'\S+\s/\S+', x):
                unittest = x.split(' /')
                if unittest[-1] in dict_measure.keys():
                    val = DECIMAL(unittest[0])
                    unit = dict_measure.get(unittest[-1])
                    text = val + ' per ' + unit
                else:
                    val = DECIMAL(unittest[0])
                    unit = unittest[-1]
                    text = val + ' per ' + unit
                return text
            elif re.match(r'\d+[A-Za-z]+', x):
                x_text = ''
                for i in range(len(x)):
                    if x[i].isdigit() == True:
                        x_text += x[i]
                    elif x[i].isdigit() == False:
                        x_text += ' ' + x[i]
                val = CARDINAL(x_text.split(' ', 1)[0])
                unit = dict_measure.get(x_text.split(' ', 1)[1].replace(' ', ''))
                text = val + ' ' + unit

            elif re.match(r'\S+\d+[A-Za-z]+', x):
                x_text = ''
                for i in range(len(x)):
                    if x[i].isdigit() == True:
                        x_text += x[i]
                    elif x[i].isdigit() == False:
                        x_text += ' ' + x[i]
                val_text = x_text.split(' ', 2)
                val = val_text[0] + val_text[1]
                val = DECIMAL(val)
                unit = dict_measure.get(x_text.split(' ', 2)[-1].replace(' ', ''))
                text = val + ' ' + unit
            elif re.match(r'\S+\d+\s[A-Za-z]+', x):
                if x.split(' ', 1)[1] in dict_measure.keys():
                    val = x.split(' ', 1)[0]
                    if val.isdigit() == True:
                        val = CARDINAL(val)
                    elif val.isdigit() == False:
                        val = DECIMAL(val)
                    unit = dict_measure.get(x.split(' ', 1)[1])
                    text = val + ' ' + unit
                else:
                    val = x.split(' ', 1)[0]
                    if val.isdigit() == True:
                        val = CARDINAL(val)
                    elif val.isdigit() == False:
                        val = DECIMAL(val)
                    text = val + ' ' + x.split(' ', 1)[1]

            return text

    except:
        return x

def ELECTRONIC(x):
        try:
            key = x.replace('.',' dot ').replace('/',' slash ').replace('-',' dash ').replace(':',' colon ').replace('_',' underscore ').replace('#',' hash tag ').replace('~',' tilde ')
            key = key.split()
            lis2 = ['dot','slash','dash','colon']

            for j in range(len(key)):
                if key[j] not in lis2: key[j]=" ".join(key[j])
            trans = " ".join(key)
            text = ''
            for i in range(len(trans)):
                if trans[i].isdigit() == True:
                    text_n = str(elec_dic.get(trans[i]))
                    text_row = ''
                    for i in range(len(text_n)):
                        if i == 0: text_row += text_n[i]
                        else: text_row += ' ' + text_n[i]
                    text += text_row
                elif trans[i].isdigit() == False:
                    text += trans[i]

        except:
            text = x

        return text.lower()


def FRACTION(x):
    try:
        x = str(x)
        y = x.split('/')
        result_string = ''
        y[0] = CARDINAL(y[0]) + ' '
        y[1] = ORDINAL(y[1])
        #if y[1] == 4:
        #    result_string = y[0] + 'quarters'
        #else:
        result_string = y[0] + ' ' + y[1] + 's'
        return(result_string)
    except:
        return(x)

def DATE(key):
    try:
        c = key.split('-')
        if len(c)==3:
            if c[1].isdigit():
                try:
                    month = dict_mon.get(str(c[1]))
                    day = dict_day.get(str(c[2]))
                    year = int(c[0])
                    if year>=2000 and year<2010: text = 'the ' + day + ' of ' + month + ' ' + str(CARDINAL(year))
                    else: text = 'the ' + day + ' of ' + month + ' ' + CARDINAL(c[0][0:2]) + ' ' + CARDINAL(c[0][2:])
                    return text.lower()
                except:
                    return key

        else:
            c = re.sub(r'[^\w]', ' ', key).split()
            if c[0].isalpha():
                try:
                    if len(c)==3:
                        month = dict_mon.get(c[0].lower())
                        day = dict_day.get(str(c[1])).replace('-',' ')
                        if int(c[2])>=2000 and int(c[2]) < 2010: text = month  + ' ' + day + ' ' + CARDINAL(c[2])
                        else: text = month  + ' ' + day + ' ' + CARDINAL(c[2][0:2]) + ' ' + CARDINAL(c[2][2:])
                        return text.lower()
                    elif len(c)==2:
                        if int(c[-1]) > 1000:
                            if int(c[1])>=2000 and int(c[1]) < 2010:
                                text = dict_mon[c[0].lower()]  + ' '+ CARDINAL(c[1])
                            else:
                                if len(c[1]) <=2: text = dict_mon[c[0].lower()] + ' ' + CARDINAL(c[1])
                                else: text = dict_mon[c[0].lower()] + ' ' + CARDINAL(c[1][0:2]) + ' ' + CARDINAL(c[1][2:])
                                return text.lower()
                            return text.lower()
                        else:
                            text = dict_mon[c[0].lower()]  + ' ' + dict_day.get(str(c[1]))
                        return text.lower()
                    else: text = key
                    return text.lower()
                except:
                    return key

            elif len(c)==1:
                if re.match(r'\S+s$', str(c[0])):
                    c[0] = c[0].replace('s','').replace(' ','')
                    if int(c[0]) >= 1000:
                        if int(c[0])>=2000 and int(c[0]) < 2010: text = re.sub(r'[\s]$', '', CARDINAL(c[0])) + 's'
                        else:
                            if int(c[0][2:]) != 0:
                                text = CARDINAL(c[0][0:2]) + ' ' + CARDINAL(c[0][2:]) + 's'
                            else: text = re.sub(r'[\s]$', '', CARDINAL(c[0])) + 's'
                        if str(text)[-2] == 'y':
                            text = text.replace('ys','ies')
                        return text.lower()
                    elif int(c[0]) < 1000 and int(c[0]) > 100:
                        ttn = str(c[0])[1] + str(c[0])[2]
                        if int(ttn)>=0 and int(ttn) < 10:
                            text = CARDINAL(c[0]) + 's'
                        else: text = CARDINAL(c[0][0]) + ' ' + CARDINAL(c[0][1:]) + 's'
                        if str(text)[-2] == 'y':
                            text = text.replace('ys','ies')
                        return text.lower()
                    else:
                        text = CARDINAL(c[0]) + 's'
                        if str(text)[-2] == 'y':
                            text = text.replace('ys','ies')
                    return text.lower()
                else:
                    if int(c[0])>=2000 and int(c[0]) < 2010: text = CARDINAL(c[0])
                    else: text = CARDINAL(c[0][0:2]) + ' ' + CARDINAL(c[0][2:])
                    return text.lower()

            else:                                           # 이제 요기
                if len(c) == 2:
                    text = 'the '+ dict_day.get(str(c[0])) + ' of ' + dict_mon.get(c[1].lower())
                c = re.sub(r'[^\w]', ' ', key).split()
                if len(c)==2:
                    try:
                        if re.match(r's$', str(c[1])):
                            text = CARDINAL(c[0]) +'s'
                        return text.lower()
                    except:
                        return key
                else:
                    try:
                        month = dict_mon.get(c[1].lower())
                        day = dict_day.get(str(c[0])).replace('-',' ')
                        if int(c[2])>=2000 and int(c[2]) < 2010:
                            text = 'the '+ day + ' of ' + month  + ' ' + CARDINAL(c[2])
                        else:
                            text = 'the '+ day + ' of ' + month  + ' ' + CARDINAL(c[2][0:2]) + ' ' + CARDINAL(c[2][2:])
                        return text.lower()
                    except:
                        return key
                return text.lower()
    except:
        return(key)

def TIME(key):
    try:
        key = key.lower()
        c = re.sub(r'[^\w]', ' ', key).split()
        if re.match(r'[^\d]', c[-1]):
            if c[-1] == 'm':
                if len(c) == 4:
                    time = CARDINAL(c[0])
                    minutes = CARDINAL(c[1])
                    if c[-2] == 'a':
                        full_time = time + ' ' + minutes + ' am'
                    elif c[-2] == 'p':
                        full_time = time + ' ' + minutes + 'pm'
                    return full_time.lower()

                elif len(c) == 3:
                    if c[-2] == 'a':
                        time = CARDINAL(c[0])
                        full_time = time + ' am'
                    elif c[-2] == 'p':
                        time = CARDINAL(c[0])
                        full_time = time + ' pm'

                    return full_time.lower()
            elif c[-1] == 'am':
                if len(c) == 3:
                    time = CARDINAL(c[0])
                    minutes = CARDINAL(c[1])
                    full_time = time + ' ' + minutes + ' am'
                elif len(c) == 2:
                    time = CARDINAL(c[0])
                    full_time = time + ' am'
                return full_time.lower()

            elif c[-1] == ' pm':
                if len(c) == 3:
                    time = CARDINAL(c[0])
                    minutes = CARDINAL(c[1])
                    full_time = time + ' ' + minutes + ' pm'
                elif len(c) == 2:
                    time = CARDINAL(c[0])
                    full_time = time + ' pm'
                return full_time.lower()

        elif re.match(r'\d', c[-1]):
            if len(c) == 3:
                time = CARDINAL(c[0])
                minutes = CARDINAL(c[1])
                seconds = CARDINAL(c[2])
                full_time = time + ' ' + minutes + ' point ' + seconds + ' seconds'
            elif len(c) == 2:
                if re.match(r'\d+am',c[-1]):
                    time = CARDINAL(re.sub(r'\D', '', c[0]))
                    minutes = CARDINAL(c[-1].split('a')[0])
                    full_time = time + ' ' + minutes + ' am'
                elif re.match(r'\d+pm',c[-1]):
                    time = CARDINAL(re.sub(r'\D', '', c[0]))
                    minutes = CARDINAL(c[-1].split('p')[0])
                    full_time = time + ' ' + minutes + ' pm'
                else:
                    time = CARDINAL(c[0])
                    minutes = CARDINAL(c[1])
                    full_time = time + minutes
            elif len(c) == 1:
                if re.match(r'\d+am',c[0]):
                    time = CARDINAL(re.sub(r'\D', '', c[0]))
                    full_time = time + ' am'

                elif re.match(r'\d+pm',c[0]):
                    time = CARDINAL(re.sub(r'\D', '', c[0]))
                    full_time = time + ' pm'
            return full_time.lower()
    except:
        full_time = key
        return full_time.lower()

def ORDINAL(key):
    c = re.sub(r'[^\w]', ' ', key).split()
    if len(c) == 1:
        if re.match(r'[\d]', c[0]):
            ord = re.sub(r'\D', '', c[0])
            ord = str(ord)
            if len(ord) == 1:
                full_ord = dict_ord.get(int(ord))
                return full_ord
            elif len(ord) == 2:
                if(int(int(ord) / 10)) == 1:
                    full_ord = dict_ord.get(int(ord))
                elif(int(int(ord) / 10) > 1):
                    t = CARDINAL(int(ord[0]+'0'))
                    if int(ord[1]) != 0:
                        o = dict_ord.get(int(ord[1]))
                        full_ord = t + ' ' + o
                    else: full_ord = dict_ord.get(int(ord[0]+'0'))
                    return full_ord
                return full_ord
            elif len(ord) == 3:
                h = CARDINAL(int(ord[0]))
                if(int(int(ord[1]+ord[2]) / 10)) == 1:
                    full_ord = h + ' hundred ' + dict_ord.get(int(ord[1]+ord[2]))
                elif(int(int(ord[1]+ord[2]) / 10) > 1):
                    if int(ord[2]) == 0:
                        full_ord = h + ' hundred ' + dict_ord.get(int(ord[1]+ord[2]))
                    else:
                        t = CARDINAL(int(ord[1]+'0'))
                        o = dict_ord.get(int(ord[2]))
                        full_ord = h + ' hundred ' + t + o
                elif(int(int(ord[1]+ord[2]) / 10) == 0):
                    if int(ord[2]) == 0:
                        full_ord = h + ' hundredth'
                    else:
                        full_ord = h + ' hundred ' + dict_ord.get(int(ord[2]))
                    return full_ord
                return full_ord
            elif len(ord) == 4:
                th = CARDINAL(int(ord[0]))
                if int(ord[1]) == 0:
                    # hundred die
                    if(int(int(ord[2]+ord[3]) / 10)) == 1:
                        full_ord = th + ' thousand ' + dict_ord.get(int(ord[2]+ord[3]))
                    elif(int(int(ord[2]+ord[3]) / 10) > 1):
                        if int(ord[3]) == 0:
                            full_ord = th + ' thousand ' + dict_ord.get(int(ord[2]+ord[3]))
                        else:
                            t = CARDINAL(int(ord[2]+'0'))
                            o = dict_ord.get(int(ord[3]))
                            full_ord = th + ' thousand ' + t + o
                        return full_ord
                    elif(int(int(ord[2]+ord[3]) / 10) == 0):
                        if int(ord[3]) == 0:
                            full_ord = th + ' thousandth'
                        else:
                            full_ord = th + ' thousand ' + dict_ord.get(int(ord[3]))
                        return full_ord
                    return full_ord
                else:
                    h = CARDINAL(int(ord[1]))
                    if(int(int(ord[2]+ord[3]) / 10)) == 1:
                        full_ord = th + ' thousand ' + h + ' hundred ' + dict_ord.get(int(ord[2]+ord[3]))
                    elif(int(int(ord[2]+ord[3]) / 10) > 1):
                        if int(ord[3]) == 0:
                            full_ord = th + ' thousand ' + h + ' hundred ' + dict_ord.get(int(ord[2]+ord[3]))
                        else:
                            t = CARDINAL(int(ord[2]+'0'))
                            o = dict_ord.get(int(ord[3]))
                            full_ord = th + ' thousand ' + h + ' hundred ' + t + o
                        return full_ord
                    elif(int(int(ord[2]+ord[3]) / 10) == 0):
                        if int(ord[3]) == 0:
                            full_ord = th + ' thousand ' + h + ' hundredth'
                        else:
                            full_ord = th + ' thousand ' + h + ' hundred ' + dict_ord.get(int(ord[3]))
                        return full_ord
                    return full_ord
            
    elif re.match(r'[^\d]', c[0]):
        full_ord = dict_ord.get(c[0])
        return full_ord

    elif re.match(r'[\d]', c[0]) and len(c) == 2:
        front_ord = CARDINAL(c[0])
        full_ord = front_ord + ' ' + c[1]
        return full_ord

def DIGIT(x):
    try:
        x = str(x)
        full_num = re.sub('[^0-9]', '', x)
        digits = []
        for num in full_num:
            digits.append(CARDINAL(num).replace("zero", "o"))
        return ' '.join(digits)
    except:
        return x

def LETTERS(x):
    try:
        lettersOnly = re.sub('[^\w]', '', x)
        if (lettersOnly[-1] == 's'):
            if (lettersOnly[-2] != lettersOnly[-2].lower):

                letterList = []
                for letter in lettersOnly[0:len(lettersOnly)-2]:
                    letterList.append(letter.lower())
                lastTwo = lettersOnly[-2].lower() + "'" + lettersOnly[-1]
                letterList.append(lastTwo)
            else: pass
            return ' '.join(letterList)
        else:

            if len(lettersOnly) == 1:
                letterLine = lettersOnly.lower()
            else:
                letterList = []
                for letter in lettersOnly:
                    letterList.append(letter.lower())
                letterLine = ' '.join(letterList)
            return letterLine
    except:
        return x

def TELEPHONE(x):
    try:
        telNum = []
        for i in range(0,len(x)):
            if re.match('[0-9]+', x[i]): telNum.append(CARDINAL(x[i]))
            elif telNum[-1] != 'sil': telNum.append('sil')
        return ' '.join(telNum)
    except: return x

for i in range(len(test_data)):
    if test_data[i,0] == 'PLAIN': test_data[i,2] = test_data[i,1]

    elif test_data[i,0] == 'PUNCT': test_data[i,2] = test_data[i,1]

    elif test_data[i,0] == 'VERBATIM': test_data[i,2] = VERBATIM(test_data[i,1])

    elif test_data[i,0] == 'CARDINAL': test_data[i,2] = CARDINAL(test_data[i,1])

    elif test_data[i,0] == 'DECIMAL': test_data[i,2] = DECIMAL(test_data[i,1])

    elif test_data[i,0] == 'FRACTION': test_data[i,2] = FRACTION(test_data[i,1])

    elif test_data[i,0] == 'DATE': test_data[i,2] = DATE(test_data[i,1])
    
    elif test_data[i,0] == 'TIME': test_data[i,2] = TIME(test_data[i,1])

    elif test_data[i,0] == 'ORDINAL': test_data[i,2] = ORDINAL(test_data[i,1])

    elif test_data[i,0] == 'LETTERS': test_data[i,2] = LETTERS(test_data[i,1])

    elif test_data[i,0] == 'TELEPHONE': test_data[i,2] = TELEPHONE(test_data[i,1])

    elif test_data[i,0] == 'DIGIT': test_data[i,2] = DIGIT(test_data[i,1])

    elif test_data[i,0] == 'ELECTRONIC': test_data[i,2] = ELECTRONIC(test_data[i,1])

    elif test_data[i,0] == 'MEASURE': test_data[i,2] = MEASURE(test_data[i,1])


for i in range(len(test_df_array)):
    id_text = ''
    id_text = str(test_df_array[i,0]) + '_' + str(test_df_array[i,1])
    test_df_array[i,0] = id_text
final_id_row = test_df_array[:,0].reshape(-1,1)
test_data = test_data[:,-1].reshape(-1,1)

final_data = np.concatenate((final_id_row, test_data), axis=1)

submission = pd.DataFrame(final_data, columns=['id','after'])
submission.to_csv("submission_upgrade.csv", index=False)