
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as DT
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def tumor_first_diag(tumor):
    tumor = tumor.sort_values(['ANON_ID', 'DATEDX'])
    tumor_fdiag = tumor.drop_duplicates(subset = ['ANON_ID'],keep='first')

    tumor_fdiag = tumor_fdiag[['ANON_ID','DATEDX']]
    tumor_fdiag = tumor_fdiag.rename(columns={"DATEDX":"FIRST_DIAG"})
    tumor_fdiag = pd.merge(tumor,tumor_fdiag,on="ANON_ID",how="inner")
    tumor_fdiag = pd.merge(tumor_fdiag,patients_s,on="ANON_ID",how="inner")
    return tumor_fdiag

def encounter_last(encounter):
    encounter_last = encounter.sort_values(['ANON_ID','APPT_WHEN'])
    encounter_last = encounter_last.drop_duplicates(subset = ['ANON_ID'],keep='last')
    encounter_last = encounter_last.rename(columns={"APPT_WHEN":"LAST_ENCOUNTER_DATE"})
    
def merge_mortality(tumor_fdiag,encounter):
    tumor_mortality = pd.merge(tumor_fdiag,encounter_last,on="ANON_ID",how="inner")
    tumor_mortality['FIVE_YR_AFTER'] = tumor_mortality['FIRST_DIAG'] + pd.DateOffset(years=5)
    tumor_mortality["FIVE_YR_MORTALITY"] = np.where((tumor_mortality["DEATH_DATE"]>tumor_mortality["FIVE_YR_AFTER"]),1,np.where((tumor_mortality["LAST_ENCOUNTER_DATE"]>tumor_mortality["FIVE_YR_AFTER"]),0,2))
    tumor_mortality = tumor_mortality[tumor_mortality["FIVE_YR_MORTALITY"]!=2]
    return tumor_mortality

def age_buckets_1(tumor_mortality):
    now = pd.Timestamp(DT.datetime.now())
    tumor_mortality1 = tumor_mortality.drop_duplicates(subset = ['ANON_ID'],keep='first')
    tumor_mortality1['AGE'] = (now - tumor_mortality1['BIRTH_DATE']).astype('<m8[Y]') 
    tumor_mortality1['BELOW25'] = np.where((tumor_mortality1['AGE'] < 25),1,0)
    tumor_mortality1['25TO30'] = np.where((25<=tumor_mortality1['AGE'])&(tumor_mortality1['AGE'] <30),1,0)
    tumor_mortality1['30TO40'] = np.where((30<=tumor_mortality1['AGE'])&(tumor_mortality1['AGE'] <40),1,0)
    tumor_mortality1['40TO50'] = np.where((40<=tumor_mortality1['AGE'])&(tumor_mortality1['AGE'] <50),1,0)
    tumor_mortality1['50TO60'] = np.where((50<=tumor_mortality1['AGE'])&(tumor_mortality1['AGE'] <60),1,0)
    tumor_mortality1['60TO75'] = np.where((60<=tumor_mortality1['AGE'])&(tumor_mortality1['AGE'] <75),1,0)
    tumor_mortality1['ABOVE75'] = np.where((75<=tumor_mortality1['AGE']),1,0)
    return tumor_mortality1


def one_hot(tumor_mortality1):
    rem = ['ANON_ID', 'TUMOR_ID', 'DATEDX', 'YEARDX','DATESTAT','DATEOFMULTIPLETUMORS','BIRTH_DATE','DEATH_DATE','LAST_ENCOUNTER_DATE','HOSP_ADMSN_TIME','HOSP_DISCH_TIME','FIVE_YR_AFTER','FIRST_DIAG','APPT_TYPE','VISIT_TYPE','AGE','BELOW25','25TO30','30TO40','40TO50','50TO60','60TO75','ABOVE75','PT_CLASS','ENC_TYPE','APPT_STATUS']
    droplist = list(rem)
    non_one_hot = list(tumor_mortality1)
    for i in rem:
        if i in non_one_hot:
            non_one_hot.remove(i)
        else:
            droplist.remove(i)
    tumor_mortality1.drop(droplist,axis=1, inplace=True)
    tumor_mortality1_hot = pd.get_dummies(tumor_mortality1,columns = non_one_hot)
    return tumor_mortality1_hot

def data_labels(tumor_mortality1_hot):
    tumor_mortality1_y = tumor_mortality1_hot.copy()
    tumor_mortality1_x = tumor_mortality1_hot.copy()
    tumor_mortality1_x.drop(['FIVE_YR_MORTALITY'], axis=1, inplace=True)
    tumor_mortality1_y=tumor_mortality1_y["FIVE_YR_MORTALITY"]
    return tumor_mortality1_x,tumor_mortality1_y

def regression(tumor_mortality1_x,tumor_mortality1_y):
    X_train, X_test, y_train, y_test = train_test_split(tumor_mortality1_x, tumor_mortality1_y, test_size=0.2)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)
    auc = metrics.roc_auc_score(y_test,y_prob[:,1])
    
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print("auc",auc)

    confusion_matrix = confusion_matrix(y_test, y_pred)
    print ("Confusion matrix:")
    print(confusion_matrix)

    feature_importance = abs(classifier.coef_[0])
    print("Feature importance:")
    print(feature_importance)

if __name__ == '__main__':
    patients_s = pd.read_csv('../stanford-oncoshare-data/MIT_V4_S_PATIENT.csv',parse_dates=['BIRTH_DATE','DEATH_DATE'])
    tumor = pd.read_csv('../stanford-oncoshare-data/MIT_V4_S_CCR_TUMOR.csv',parse_dates=['DATEDX'])
    encounter = pd.read_csv('../stanford-oncoshare-data/MIT_V4_S_ENCOUNTER.csv',parse_dates=['APPT_WHEN'])

    tumor_fdiag = tumor_first_diag(tumor)
    tumor_mortality = merge_mortality(tumor_fdiag)
    encounter_last = encounter_last(encounter) 
    tumor_mortality = merge_mortality(tumor_fdiag,encounter) 
    tumor_mortality1 = age_buckets_1(tumor_mortality)
    tumor_mortality1_hot = one_hot(tumor_mortality1)
    tumor_mortality1_x,tumor_mortality1_y = data_labels(tumor_mortality1_hot)
    regression(tumor_mortality1_x,tumor_mortality1_y)

