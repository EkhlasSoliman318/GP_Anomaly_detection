

import numpy as np
import pandas as pd
from pathlib import Path
import glob
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm_notebook
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_confusion_matrix
#from rgf.sklearn import RGFRegressor
import pickle


def preprocessing(df):
    cols = ["Unnamed: 0", 'Unnamed: 0.1']
    df.drop(columns=cols, inplace=True)
    dropped = ['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value', 'FC3_Read_Holding_Register', 'FC4_Read_Coil',
               '_c0', 'current_temperature', 'date', 'door_state', 'fridge_temperature', 'humidity', 'label',
               'latitude', 'light_status', 'longitude', 'motion_status', 'pressure', 'sphone_signal', 'temp_condition',
               'temperature', 'thermostat_status', 'time', 'type']
    df.drop(columns=dropped, inplace=True)
    cols = ["src_ip", "src_port", "dst_ip", "dst_port"]
    df.drop(columns=cols, inplace=True)
    cols = [
        'dst_pkts','http_method','http_orig_mime_types','http_request_body_len','http_resp_mime_types','http_response_body_len','http_status_code','http_trans_depth','http_uri','http_user_agent','http_version','missed_bytes','src_bytes','src_ip_bytes','ssl_issuer','ssl_resumed','ssl_subject','ssl_version','weird_addl',
    ]
    df.drop(columns=cols, inplace=True)
    ####################################
    cols = ['device',
            'dns_qclass',
            'dst_bytes',
            'ssl_cipher',
            'ssl_established',
            'weird_name',
            'weird_notice']
    df.drop(columns=cols, inplace=True)
    ###################################

    df.dropna(inplace=True)

    #all_data = pd.concat([data, to_be_tested])


    categorical = [val for val in df.columns if df[val].dtype == "object"]
    for i in categorical[:-1]:
        label_encoder(df, i)

    scaler = RobustScaler()
    scaler.fit(df.drop(columns=["nw_label", "nw_type"]))
    scaled_cols = scaler.transform(df.drop(columns=["nw_label", "nw_type"]))
    features = df.columns[:-2]
    data_scaled = pd.DataFrame(scaled_cols, columns=features)
    x = data_scaled.copy()
    y = df["nw_label"].astype("int64")

    return x,y

def label_encoder(df,a):

    le = LabelEncoder()
    df[a] = le.fit_transform(df[a])

data=pd.read_csv("E:/اخر داتا ان شاء الله/aggregated/training.csv")
train_x,train_y=preprocessing(data)
sampling = SMOTETomek(random_state=42)
X_resambled,Y_resambled=sampling.fit_resample(train_x,train_y)


model1=RandomForestClassifier(random_state=42)
model1.fit(X_resambled,Y_resambled)
pickle.dump(model1, open('E:/اخر داتا ان شاء الله/aggregated/RF_model.pkl', 'wb'))

#
#
# xgboost_model = XGBClassifier(random_state=42)
# xgboost_model.fit(X_train, y_train)
# y_pred_xg = xgboost_model.predict(X_test)
# f_xg = f1_score(y_test, y_pred_xg, average='macro')
# acc_xg = accuracy_score(y_test, y_pred_xg)
# acc_xg
#
#
# # In[21]:
#
#
# print(classification_report(y_test, y_pred_xg))
# confusion_matrix = confusion_matrix(y_test, y_pred_xg)
# ConfusionMatrixDisplay(confusion_matrix, display_labels = xgboost_model.classes_).plot()
#
#
# # In[23]:
#
#
# pickle.dump(xgboost_model, open('E:/cs project/elg7186-project-group_project_-5-main/elg7186-project-group_project_-5-main/models/xgboost_model.pkl', 'wb'))
#
#
# # In[ ]:




