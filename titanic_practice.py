#coding=utf-8
"""
Practice ML for project "Titanic"
@Author: Arkuyo

"""
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# 調整長度
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

# 資料位置
train = pd.read_csv("D:\Anthony\Titanic\\train.csv")
test = pd.read_csv("D:\Anthony\Titanic\\test.csv")
submit = pd.read_csv('D:\Anthony\Titanic\\gender_submission.csv')

# 資料合併
data = train.append(test)
data.reset_index(inplace=True, drop=True)

# 家庭組成=帶父母+帶兄弟姊妹
data['Family_Size'] = data['Parch'] + data['SibSp']
data['Title1']= data['Name'].str.split(", ", expand=True)[1]
data['Name'].str.split(", ", expand=True).head(3)
data['Title1']= data['Title1'].str.split(".", expand=True)[0]
data['Title1'].head(3)
data['Title1'].unique()

# 將稱謂較少的合併成主流
data.groupby(['Title1'])['Age'].mean()
data['Title2']= data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Capt','Col','Don','Dona','Jonkheer','Rev','Sir'],
                                       ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mrs','Mr','Mr','Mr'])
data['Title2'].unique()
#pd.crosstab(data['Title2'],data['Survived']).T.style.background_gradient(cmap='summer_r')

# 將票資訊用單一英文字母表示，數字則用X代替
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
data['Ticket_info'].unique()

# 登船港口空白者用S港口填滿
data['Embarked'] = data['Embarked'].fillna('S')

# 票價空白者用平均值填滿
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# 客艙空白者用'NoCabin'表示
data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
data["Cabin"].unique()

# 將類別資料轉為整數
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Title1'] = data['Title1'].astype('category').cat.codes
data['Title2'] = data['Title2'].astype('category').cat.codes
data['Cabin'] = data['Cabin'].astype('category').cat.codes
data['Ticket_info'] = data['Ticket_info'].astype('category').cat.codes

# 使用隨機森林來推測年齡
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))
                     ]
rfModel_age = RandomForestRegressor(n_estimators=2000, random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2', 'Cabin', 'Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X=dataAgeNull[ageColumns])
dataAgeNull.loc[:, "Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)

dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
#dataTrain.columns
dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2', 'Ticket_info', 'Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2', 'Ticket_info', 'Cabin']]
#dataTrain



# 載入隨機森林演算法(Random Forest)來預測存活率
rf = RandomForestClassifier(criterion='gini',
                            n_estimators=1000,
                            min_samples_split=12,
                            min_samples_leaf=1,
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)

pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']),
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])),
          axis = 1).sort_values(by='importance', ascending = False)[:20]

rf_res = rf.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)
submit











