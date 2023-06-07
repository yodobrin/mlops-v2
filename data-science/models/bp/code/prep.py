# Import libraries
import os
import argparse
from azureml.core import Run
import pandas as pd
import numpy as np
import warnings
#ignore warnings 
warnings.filterwarnings('ignore') 
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from functools import reduce

from IPython.display import display
pd.options.display.max_columns = None

from azureml.fsspec import AzureMachineLearningFileSystem
import mlflow
import mlflow.sklearn

def main():
    """Main function of the script."""

    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--prepped_data", type=str, dest='prepped_data', default='prepped_data', help="path to prepped data")
    
#     parser.add_argument("--input-individual-customer-portfolio-data", type=str, dest='raw_dataset_id', help='raw dataset')
#     parser.add_argument("--input-policy-base-layer-data", type=str, dest='raw_dataset_id', help='raw dataset')
    args = parser.parse_args()
    save_folder = args.prepped_data

    # Start Logging
    mlflow.start_run()
    # Get the experiment run context
    run = Run.get_context()
    
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)
    
    # load the data (passed as an input dataset)
    print("Loading Data...")
#     INDIVIDUAL_ENTITY_DATA = run.input_datasets['indiviual_entity_raw_data'].to_pandas_dataframe()
#     INDIVIDUAL_CUSTOMER_PORTFOLIO = run.input_datasets['individual_customer_portfolio_raw_data'].to_pandas_dataframe()
#     POLICY_BASE_LAYER = run.input_datasets['policy_base_layer_raw_data'].to_pandas_dataframe()
    INDIVIDUAL_ENTITY_DATA = pd.read_csv(args.data+'cursor_INDIVIDUAL_ENTITY_DATA.csv')
    INDIVIDUAL_CUSTOMER_PORTFOLIO =pd.read_csv(args.data+'cursor_INDIVIDUAL_CUSTOMER_PORTFOLIO.csv')

    print("INDIVIDUAL_ENTITY_DATA shape : ",INDIVIDUAL_ENTITY_DATA.shape)
    print("INDIVIDUAL_CUSTOMER_PORTFOLIO shape : ",INDIVIDUAL_CUSTOMER_PORTFOLIO.shape)
    print("POLICY_BASE_LAYER shape : ",POLICY_BASE_LAYER.shape)

    df_mid = pd.merge(INDIVIDUAL_CUSTOMER_PORTFOLIO, INDIVIDUAL_ENTITY_DATA, on='ENTITY_ID', how='inner')
    POLICY_BASE_LAYER.rename(columns={'POLICY_HEADER_ID': 'ENTITY_ID'}, inplace=True)
    df = pd.merge(df_mid, POLICY_BASE_LAYER, on='ENTITY_ID', how='inner')
    #proposal=0  and policy = 1
    df_final = df[(df['POLICY_STATUS']=='Policy') | (df['POLICY_STATUS']=='Proposal')]
    df_final['POLICY_STATUS'] = np.where(df_final['POLICY_STATUS']=='Policy',1,0)
    df_final.to_csv('Buying_propensity_Data.csv',index=False)
    # Identify numeric and catagorical variables
    df_new = df_final.copy()
    Column_list = df_new.columns.tolist()
    Column_list_cat = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
    Column_list_num = df_new[df_new.columns.difference(['POLICY_STATUS'])].select_dtypes(include=['float64', 'int64']).columns.tolist()
    data_explain = pd.DataFrame(columns = ["column"])
    data_explain['column'] = Column_list
    # Removing variables where more than 30% is blank or null
    dict_1={}
    for i in Column_list:
        percent_missing = (df_new[i].isnull().sum()/len(df_new))* 100
        if percent_missing >= 30:
            dict_1[i] = 1
            df_new.drop([i], axis = 1,inplace=True)
        else:
            dict_1[i] = 0
    data_explain["missing_grt30"]=data_explain.column.astype(str).map(dict_1)

    Column_list_num = df_new[df_new.columns.difference(['POLICY_STATUS'])].select_dtypes(include=['float64', 'int64']).columns.tolist()
    Column_list = df_new.columns.tolist()
    Column_list_cat = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
    Column_list_num = df_new[df_new.columns.difference(['POLICY_STATUS'])].select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Missing values for numeric variables by mean 
    for i in Column_list_num:
        df_new[i].fillna((df_new[i].mean()), inplace=True)

    # Missing values for catagorical variables by mode
    for i in Column_list_cat:
        df_new[i].fillna(df_new[i].mode().iloc[0], inplace=True)
        df_new[i] = df_new[i].astype('object')


    # Outlier Treatment using IQR method
    for i in Column_list_num:
        df_new.sort_values(by=i, inplace=True)
        filtered_vals = df_new[i][~np.isnan(df_new[i])]

        q1 = np.percentile(filtered_vals, 25)         
        q3 = np.percentile(filtered_vals, 75)          
        IQR = q3-q1  

        lower_limit = q1 -(1.5 * IQR) 

        if lower_limit < min(df_new[i]):
            lower_limit = min(df_new[i])

        upper_limit = q3 +(1.5 * IQR)

        df_new.loc[df_new[i] >= upper_limit, i] = upper_limit
        df_new.loc[df_new[i] <= lower_limit, i] = lower_limit 

    #variable selection
    df_backup = df_new.copy()

    # creating instance of labelencoder
    labelencoder = LabelEncoder()

    # Encoding the categorical variable
    cat = df_new.select_dtypes(include=['object']).astype('str').apply(lambda x: labelencoder.fit_transform(x))
    num = df_new.select_dtypes(include=['float64', 'int64','int32'])

    df_new = cat.merge(num, left_index=True, right_index=True)
    features = df_new[df_new.columns.difference(['POLICY_STATUS'])]
    labels = df_new['POLICY_STATUS']

    #Random Forest
    clf = RandomForestClassifier()

    clf.fit(features,labels)

    preds = clf.predict(features)


    accuracy = accuracy_score(preds,labels)


    VI = DataFrame(clf.feature_importances_, columns = ["RF"], index=features.columns)
    VI = VI.reset_index()

    #Recursive feature elimination

    model = LogisticRegression()
    rfe = RFE(estimator=model, n_features_to_select=20)
    fit = rfe.fit(features, labels)


    Selected = DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)
    Selected = Selected.reset_index()

    # Extratrees Classifier

    model = ExtraTreesClassifier()
    model.fit(features, labels)
    FI = DataFrame(model.feature_importances_, columns = ["Extratrees"], index=features.columns)
    FI = FI.reset_index()

    # Chi Square

    df1 = df_new.copy()
    d = defaultdict(preprocessing.LabelEncoder)

    # Encoding the categorical variable
    fit = df1.apply(lambda x: d[x.name].fit_transform(x))

    #Convert the categorical columns based on encoding
    for i in list(d.keys()):
        df1[i] = d[i].transform(df1[i])

    features1 = df1[df1.columns.difference(['POLICY_STATUS'])]
    labels1 = df1['POLICY_STATUS']



    model = SelectKBest(score_func=chi2, k=5)
    fit = model.fit(features1, labels1)

    pd.options.display.float_format = '{:.2f}'.format
    chi_sq = DataFrame(fit.scores_, columns = ["Chi_Square"], index=features1.columns)
    chi_sq = chi_sq.reset_index()

    # L1 feature selection

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
    model = SelectFromModel(lsvc,prefit=True)

    l1 = DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
    l1 = l1.reset_index()

    #Combine
    dfs = [VI, Selected, FI, chi_sq, l1]
    final_results = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs)

    #Variable Score
    columns = ['RF', 'Extratrees', 'Chi_Square']

    score_table = pd.DataFrame({},[])
    score_table['index'] = final_results['index']

    for i in columns:
        score_table[i] = final_results['index'].isin(list(final_results.nlargest(5,i)['index'])).astype(int)

    score_table['RFE'] = final_results['RFE'].astype(int)
    score_table['L1'] = final_results['L1'].astype(int)

    score_table['final_score'] = score_table.sum(axis=1)
    score_table.rename(columns = {'index':'column'}, inplace = True)
    data_explain_final = pd.merge(data_explain, score_table, on='column', how='left')
    Final_Col_list = score_table['column'][score_table['final_score']>=2]
    Final_Col_list = Final_Col_list.tolist()

    print("Final_Col_list : ",Final_Col_list)
    for name in Final_Col_list:
        data_explain_final.loc[(data_explain_final['column']==name), 'Final_variables'] = 'Yes'

    data_explain_final.to_csv('Business_explainablity_Buying_propensity.csv',index=False)

    #EDA
    Final_Col_list.append('POLICY_STATUS')
    df_Model = df_backup[Final_Col_list]
    drop_list = ['ENTITY_ID','POLICY_VERSION','NUMBER_OF_ACTIVE_POLICIES', 'DATE_OF_BIRTH','ENDORS_END_DATE','ENDORS_START_DATE','LAST_UPDATE_DATE_y', 'POLICY_ID', 'POLICY_START_DATE','PROPOSAL_VALIDATION_DATE','PRODUCT_DESCRIPTION']
    df_Model.drop(columns=[col for col in drop_list if col in df_Model], inplace=True)

    #Model

    Column_list = df_Model.select_dtypes(include=['object', 'category']).columns.tolist()
    df_final = pd.get_dummies(df_Model, columns=Column_list, sparse=True)
    data_final_vars=df_final.columns.values.tolist()

    print("df_final shape : ",df_final.shape)

    print("data_final_vars : ",data_final_vars)
    # Save the prepped data
    print("Saving Data...")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder,'data.csv')
    df_final.to_csv(save_path, index=False, header=True)
    print("saved Data")
    # End the run
    run.complete()
    # Stop Logging
    mlflow.end_run()
    
if __name__ == "__main__":
    main()