import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

df_members = pd.read_csv('../input/members_v3.csv')
df_transactions = pd.read_csv('../input/transactions.csv')

print("Finished")

print("df_transactions size:")
print(df_transactions.shape)


def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and (np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and (np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and (np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)


def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)


change_datatype(df_transactions)
change_datatype_float(df_transactions)

change_datatype(df_members)
change_datatype_float(df_members)

# Pour garder transaction_date et membership_expire_date dans le bon format
df_transactions = df_transactions.assign(transaction_date_int=df_transactions['transaction_date'].values)
df_transactions = df_transactions.assign(membership_expire_date_int=df_transactions['membership_expire_date'].values)

# Feature Discount
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']

df_transactions['discount'].unique()

# Feature is_discount
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)

# Feature amount per day
df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

date_cols = ['transaction_date', 'membership_expire_date']
print(df_transactions[date_cols].dtypes)

# Convert to date
for col in date_cols:
    df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')

print("df_transactions types:")
print(df_transactions[date_cols].dtypes)
print("df_transactions row example:")
print(df_transactions.loc[12000, ['membership_expire_date', 'transaction_date']])

# Membership duration
# --- difference in days ---

df_transactions['membership_duration'] = df_transactions['membership_expire_date'].sub(
    df_transactions['transaction_date'], axis=0)
df_transactions['membership_duration'] = df_transactions['membership_duration'].values / np.timedelta64(1, 'D')
df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)

print(df_transactions.iloc[3, :])

# Problème avec les mois (la durée d'un mois varie et ça fait tout chier)
# ---difference in months ---
# df_transactions['membership_duration_M'] = df_transactions['membership_expire_date'].sub(df_transactions['transaction_date'],axis=0)
# df_transactions['membership_duration_M']= df_transactions['membership_duration_M'].values / np.timedelta64(1, 'M')
# df_transactions['membership_duration_M'] = df_transactions['membership_duration_M'].round().astype(int)


# Pas de expiration_date pour members_v3.csv !!!
# Convert date for members
date_cols = ['registration_init_time', 'expiration_date']

# for col in date_cols:
# df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')

# Registration Duration
# --- difference in days ---
# df_members['registration_duration'] = df_members['expiration_date'].sub(df_members.registration_init_time,axis=0)
# df_members['registration_duration'] = df_members['registration_duration'].values / np.timedelta64(1, 'D')
# df_members['registration_duration'] = df_members['registration_duration'].astype(int)

# ---difference in months ---
# df_members['registration_duration_M'] = (df_members['expiration_date'].sub(df_members['registration_init_time'],axis=0))/ np.timedelta64(1, 'M')
# df_members['registration_duration_M'] = df_members['registration_duration_M'].round().astype(int)

# --- Reducing and checking memory again ---
change_datatype(df_members)
change_datatype_float(df_members)

# ON NE GARDE QUE LES PLUS RECENTES TRANSACTIONS
df_transactions = df_transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
df_transactions = df_transactions.drop_duplicates(subset=['msno'], keep='first')

change_datatype(df_transactions)
change_datatype_float(df_transactions)

print("Finished")

#-- merging the two dataframes---
df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')

#Commenté pour le notebook
#--- deleting the dataframes to save memory
#del df_transactions
#del df_members

#New Feature
#df_comb['reg_mem_duration'] = df_comb['registration_duration'] - df_comb['membership_duration']
#df_comb['reg_mem_duration_M'] = df_comb['registration_duration_M'] - df_comb['membership_duration_M']

#df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)
#df_comb['autorenew_&_not_cancel'].unique()

#Feature 8
df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
df_comb['notAutorenew_&_cancel'].unique()

#Feature 9
#df_comb['long_time_user'] = (((df_comb['registration_duration'] / 365).astype(int)) > 1).astype(int)

#For memory
#datetime_cols = list(df_comb.select_dtypes(include=['datetime64[ns]']).columns)

#df_comb = df_comb.drop([datetime_cols], 1)

print(df_comb.shape)

train = pd.read_csv('/home/dsluser31/WSDM/input/train.csv')
train = pd.concat((train, pd.read_csv('/home/dsluser31/WSDM/input/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('/home/dsluser31/WSDM/input/sample_submission_v2.csv')

print(train.shape)
print(test.shape)

train = pd.merge(train,df_comb,on='msno',how='left')
test = pd.merge(test,df_comb,on='msno',how='left')

print(train.shape)
print(test.shape)


print("Finished")

df_logs = pd.read_csv('../input/user_logs_SUM.csv')

print(train.shape)
print(test.shape)

train = pd.merge(train,df_logs,on='msno',how='left')
test = pd.merge(test,df_logs,on='msno',how='left')

print(train.shape)
print(test.shape)
print("Finished")


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

#On jette aussi 'bd' car c'est le grand n'importe quoi

cols = [c for c in train.columns if c not in ['is_churn','msno','gender','bd','transaction_date','membership_expire_date']]

params = {
        'eta': 0.004, #use 0.002
        'max_depth': 30,
        'min_child_weight':1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 1,
        'silent': True,
        'nthread': 20,
        'subsample':0.8,
        'colsample_bytree': 0.8
    }
print("Features used:")
print(list(train[cols]))
x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=1)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 2000,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500

pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('submission_Arnaud.csv', index=False)