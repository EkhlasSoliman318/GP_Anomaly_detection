
from train_model import preprocessing
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

to_be_tested=pd.read_csv("E:/اخر داتا ان شاء الله/aggregated/testing.csv")

def predict_data(df):

    test_x,test_y=preprocessing(df)
    model = pickle.load(open('E:/اخر داتا ان شاء الله/aggregated/RF_model.pkl','rb'))
    preds = model.predict(test_x)
    print("precision_score : ", precision_score(test_y, preds))

    return pd.DataFrame(preds,columns=["prediction"])

predictions=predict_data(to_be_tested)
predictions.to_csv("E:/اخر داتا ان شاء الله/aggregated/predictions.csv")


