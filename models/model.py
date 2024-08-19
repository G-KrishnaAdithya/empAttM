import numpy as np 
import pandas as pd
import pickle
data=pd.read_csv('HR_Engagement_Sat_Sales.csv')
for i in data[['EMP_Sat_OnPrem_1','EMP_Sat_OnPrem_2','EMP_Sat_OnPrem_3','EMP_Sat_OnPrem_4','EMP_Sat_OnPrem_5']]:
  data[i]=data[i].fillna(data[i].mode()[0])
data['EMP_Sat_OnPrem']=data[['EMP_Sat_OnPrem_1', 'EMP_Sat_OnPrem_2', 'EMP_Sat_OnPrem_3', 'EMP_Sat_OnPrem_4', 'EMP_Sat_OnPrem_5']].mean(axis=1)
data['EMP_Sat_Remote']=data[['EMP_Sat_Remote_1', 'EMP_Sat_Remote_2', 'EMP_Sat_Remote_3', 'EMP_Sat_Remote_4', 'EMP_Sat_Remote_5']].mean(axis=1)
data['EMP_Engagement']=data[['EMP_Engagement_1', 'EMP_Engagement_2', 'EMP_Engagement_3','EMP_Engagement_4', 'EMP_Engagement_5']].mean(axis=1)
data['Emp_Work_Status']=data[['Emp_Work_Status2','Emp_Work_Status_3', 'Emp_Work_Status_4', 'Emp_Work_Status_5']].mean(axis=1)
data['Emp_Competitive']=data[['Emp_Competitive_1', 'Emp_Competitive_2', 'Emp_Competitive_3', 'Emp_Competitive_4', 'Emp_Competitive_5']].mean(axis=1)
data['Emp_Collaborative']=data[['Emp_Collaborative_1', 'Emp_Collaborative_2', 'Emp_Collaborative_3','Emp_Collaborative_4', 'Emp_Collaborative_5']].mean(axis=1)
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
for i in data[['Department', 'GEO', 'Role', 'sales', 'salary', 'Gender']]:
  data[i]=label_en.fit_transform(data[i])
data.drop(['ID', 'Name','Sensor_StepCount','Sensor_Heartbeat(Average/Min)','Sensor_Proximity(1-highest/10-lowest)'], axis=1, inplace=True)
data.drop(['Rising_Star','Critical','CSR Factor','Women_Leave','Men_Leave'], axis=1, inplace=True)
data.drop(['EMP_Sat_OnPrem_1',
       'EMP_Sat_OnPrem_2', 'EMP_Sat_OnPrem_3', 'EMP_Sat_OnPrem_4',
       'EMP_Sat_OnPrem_5', 'EMP_Sat_Remote_1', 'EMP_Sat_Remote_2',
       'EMP_Sat_Remote_3', 'EMP_Sat_Remote_4', 'EMP_Sat_Remote_5',
       'EMP_Engagement_1', 'EMP_Engagement_2', 'EMP_Engagement_3',
       'EMP_Engagement_4', 'EMP_Engagement_5','Emp_Work_Status2',
       'Emp_Work_Status_3', 'Emp_Work_Status_4', 'Emp_Work_Status_5','Emp_Competitive_1', 'Emp_Competitive_2',
       'Emp_Competitive_3', 'Emp_Competitive_4', 'Emp_Competitive_5','Emp_Collaborative_1', 'Emp_Collaborative_2',
        'Emp_Collaborative_3','Emp_Collaborative_4', 'Emp_Collaborative_5'], axis=1, inplace=True)
data.drop(['Trending Perf', 'Talent_Level', 'Validated_Talent_Level','last_evaluation'], axis=1, inplace=True)
x=data.drop('left_Company', axis=1)
y=data['left_Company']
x.drop(['EMP_Sat_OnPrem','sales','Emp_Collaborative','Emp_Work_Status','GEO','Role','Department','Will_Relocate','Gender','promotion_last_5years','Emp_Competitive','Work_accident','salary','Emp_Position','Emp_Title', 'Emp_Identity'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pickle.dump(rf ,open('model.pkl','wb'))