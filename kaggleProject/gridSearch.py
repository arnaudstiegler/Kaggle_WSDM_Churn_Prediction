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

# Convert date for members
date_cols = ['registration_init_time', 'expiration_date']



# --- Reducing and checking memory again ---
change_datatype(df_members)
change_datatype_float(df_members)

# Only keeping latest transactions
df_transactions = df_transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
df_transactions = df_transactions.drop_duplicates(subset=['msno'], keep='first')

change_datatype(df_transactions)
change_datatype_float(df_transactions)

print("Finished")

#-- merging the two dataframes---
df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')


#New Feature
df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
df_comb['notAutorenew_&_cancel'].unique()


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

#First approx for parameters

cols = [c for c in train.columns if c not in ['is_churn','msno','gender','bd','transaction_date','membership_expire_date']]

from sklearn.model_selection import GridSearchCV
param_test1 = {
 'max_depth':range(10,100,10)
}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.02, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=20, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='neg_log_loss',n_jobs=20,iid=False, cv=5)
gsearch1.fit(train[cols], train['is_churn'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

sortie = pd.from_dict(gsearch1.best_params_)
sortie.to_csv('bestParam.csv')