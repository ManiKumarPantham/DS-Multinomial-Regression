###########################################################################################
A University would like to effectively classify their students based on the program they are 
enrolled in. Perform multinomial regression on the given dataset and provide insights 
(in the documentation).
a.	prog: is a categorical variable indicating what type of program a student is in: “General” (1), “Academic” (2), or “Vocational” (3).
b.	Ses: is a categorical variable indicating someone’s socioeconomic status: “Low” (1), “Middle” (2), and “High” (3).
c.	read, write, math, and science are their scores on different tests.
d.	honors: Whether they are an honor roll or not.

###########################################################################################

# Importing required libraries into Python
import numpy as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the data into python
data = pd.read_csv('D:/Hands on/27_Multinomial Regression/Assignment/mdata.csv')

# Droping unrelated columns
data1 = data.drop(['Unnamed: 0', 'id'], axis = 1)

data1.columns

# Splting into X and Y
Y = data['prog']

X = data.loc[:, ['female', 'ses', 'schtyp', 'read', 'write', 'math', 'science', 'honors']]

# Spliting into categorical and numerical features
cat_feature = X.select_dtypes(include = object)

num_feature = X.select_dtypes(exclude = object)

# Checking for null values
num_feature.isna().sum()

# Creating a dummy variables
categorical = pd.get_dummies(cat_feature, drop_first = True)

# Concatinating two data sets
data2 = pd.concat([categorical, num_feature], axis = 1)

# Creating a MixMaxScaler object
min = MinMaxScaler()

# Scaling the data
data_scale = pd.DataFrame(min.fit_transform(data2), columns = data2.columns)

# Spliting into train and test data
x_train, x_test, y_train, y_test = train_test_split(data_scale, Y, test_size = 0.2, random_state = 0)

# Creating a object
logit = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')

# Building a model
logit_model = logit.fit(x_train, y_train)

# Prediction on test data
pred1 = logit_model.predict(x_test)

# Test data accuracy
accuracy_score(y_test, pred1)

# Prediction on train data
pred2 = logit_model.predict(x_train)

# Train data prediction
accuracy_score(y_train, pred2)


##########################################################################################################
You work for a consumer finance company which specializes in lending loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision: 
• If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company 
• If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company 

The data given below contains the information about past loan applicants and whether they ‘defaulted’4 or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. 

In this case study, you will use EDA to understand how consumer attributes and loan attributes influence the tendency of default. 

When a person applies for a loan, there are two types of decisions that could be taken by the company: 

1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below: 
o Fully paid: Applicant has fully paid the loan (the principal and the interest rate) 
o Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'. 
o Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan  
2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

This company is the largest online loan marketplace, facilitating personal loans, business loans, and financing of medical procedures. Borrowers can easily access lower interest rate loans through an online interface.  
 Like most other lending companies, lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). The credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who default cause the largest amount of loss to the lenders. In this case, the customers labelled as 'charged-off' are the 'defaulters'.  
 If one is able to identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. 
In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilize this knowledge for its portfolio and risk assessment.  

Perform Multinomial regression on the dataset in which loan_status is the output (Y) variable and it has three levels in it. 

##########################################################################################################

# Importing required libraries 
import numpy as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the data into Python
data = pd.read_csv('D:/Hands on/27_Multinomial Regression/Assignment/loan.csv')

data.columns

# Droping unrelated columns

data.drop(['id', 'member_id', 'issue_d', 'url', 'pymnt_plan', 'desc', 'purpose', 'title', 'zip_code', 
           'addr_state', 'earliest_cr_line', 'mths_since_last_delinq', 'mths_since_last_record',
           'pub_rec', 'initial_list_status', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
           'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code', 
           'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'acc_now_delinq', 
           'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
           'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 
           'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 
           'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 
           'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
           'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 
           'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 
           'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 
           'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 
           'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_il_high_credit_limit', 'emp_title', 'total_bc_limit', 
           'emp_length', 'funded_amnt', 'total_pymnt_inv', 'sub_grade'], axis = 1, inplace = True)

# Variance
data.var()

# Correlation coefficient
a = data.corr()

# Information of the dataset
data.info()

# removing the strings and converting into numerical data type
data['term'] = data.term.str.replace(' months', "").astype('int64')
data['int_rate'] = data['int_rate'].str.replace('%', "").astype('float64')
data['revol_util'] = data['revol_util'].str.replace('%', "").astype('float64')

# Auto EDA
import dtale

df = dtale.show(data)

df.open_browser()

# Checking for duplicates
data.duplicated().sum()

# CHecking for null values
data.isna().sum()

# Replacing null values iwth mean value
data.fillna(data.mean(), inplace = True)

data.isna().sum()

# Spliting the data into X and Y
Y = data['loan_status']

X = data.loc[:, data.columns != "loan_status"]

# Spliting the data into categorical and numerical features
cat_feature = X.select_dtypes(include = object)

num_feature = X.select_dtypes(exclude = object)

# Creating a dummy variables
categorical = pd.get_dummies(cat_feature, drop_first = True)

# Concatinating two datasets
df = pd.concat([num_feature, categorical], axis = 1)

# Creating a MinMaxScaler object
min = MinMaxScaler()

# Scaling the data
df_scale = pd.DataFrame(min.fit_transform(df), columns = df.columns)

# Spliting the data into train test 
x_train, x_test, y_train, y_test = train_test_split(df_scale, Y, random_state = 0, test_size = 0.2)

# Creating a object
logit = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')

# Building a model
logit_model = logit.fit(x_train, y_train)

# Prediction on test data
test_pred = logit_model.predict(x_test)

# Test data accuracy
accuracy_score(y_test, test_pred)

# Prediction on train data
train_pred = logit_model.predict(x_train)

# Train data accuracy
accuracy_score(y_train, train_pred)

