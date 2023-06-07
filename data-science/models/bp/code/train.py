import argparse
import os

# importing necessary libraries
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import joblib

import mlflow
import mlflow.sklearn

from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.tracking import MlflowClient

class ModelWrapper(PythonModel):
    def __init__(self, model):
        self._model = model

    def predict(self, context: PythonModelContext, data):
        # You don't have to keep the semantic meaning of `predict`. You can use here model.recommend(), model.forecast(), etc
        # transform 
        pred = self._model.predict_proba(data)[:,1]
        print(f"predict: {self._model.predict_proba(data)}, pred:{pred}")

        return pred.tolist()

    # You can even add extra functions if you need to. Since the model is serialized,
    # all of them will be available when you load your model back.
    def predict_batch(self, data):
        pass
    
input_sample = pd.DataFrame(
    data=[
        {
            "AUD_SUB_BATCH_ID_y": 47.0,
            "CALENDAR_YEAR": 2022.0,
            "NUMBER_OF_POLICY_PURCHASED_IN_PAST": 0.0,
            "POLICIES_PURCHASED_IN_LAST_24_MONTHS": 0.0,
            "POLICIES_PURCHASED_IN_LAST_36_MONTHS": 0.0,
            "BUSINESS_TYPE_DESC_NB Contract": 0.0,
            "BUSINESS_TYPE_DESC_NB Endorsment Contract": 0.0,
            "BUSINESS_TYPE_DESC_NB Endorsment Proposal": 0.0,
            "BUSINESS_TYPE_DESC_NB Proposal": 1.0,
            "BUSINESS_TYPE_DESC_NB Proposal Endorsment": 0.0,
            "BUSINESS_TYPE_DESC_Renewal Contract": 0.0,
            "BUSINESS_TYPE_DESC_Renewal Contract Endorsment": 0.0,
            "BUSINESS_TYPE_DESC_Renewal Endorsment Proposal": 0.0,
            "BUSINESS_TYPE_DESC_Renewal Proposal": 0.0,
            "BUSINESS_TYPE_DESC_Renewal Proposal Endorsment": 0.0,
            "PAYMENT_CHANEL_METHOD_Conversion": 0.0,
            "PAYMENT_CHANEL_METHOD_IDIT_UI": 1.0,
            "PAYMENT_CHANEL_METHOD_Interface": 0.0,
            "PAYMENT_CHANEL_METHOD_Original Conversion": 0.0,
            "PREFERRED_DELIVERY_TYPE_Do Not Dispatch": 1.0,
            "PREFERRED_DELIVERY_TYPE_Email": 0.0,
            "PREFERRED_DELIVERY_TYPE_Fax": 0.0,
            "PREFERRED_DELIVERY_TYPE_Registered mail": 0.0,
            "PREFERRED_DELIVERY_TYPE_Regular Mail": 0.0,
            "PREFERRED_DELIVERY_TYPE_Special Delivery": 0.0
    }
    ]
)

output_sample = pd.DataFrame(data=[{"class": 0, "confidence": 0.25}])

from mlflow.models.signature import infer_signature

# signature = infer_signature(input_sample, output_sample)

def main():
    
    parser = argparse.ArgumentParser()

#     parser.add_argument('--kernel', type=str, default='linear',
#                         help='Kernel type to be used in the algorithm')
#     parser.add_argument('--penalty', type=float, default=1.0,
#                         help='Penalty parameter of the error term')
    #parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")

    # Start Logging
    mlflow.start_run()

    # enable autologging
#     mlflow.sklearn.autolog()
    mlflow.sklearn.autolog(log_models=False)
    args = parser.parse_args()
    training_data = args.train_data
    
    mlflow.log_param('train_data', str(args.train_data))
#     mlflow.log_metric('Penalty', float(args.penalty))

    # load the prepared data file in the training folder
    print("Loading Data...")
    file_path = os.path.join(training_data,'data.csv')
    df_final = pd.read_csv(file_path)
    data_final_vars=df_final.columns.values.tolist()
    
    Y_Var=['POLICY_STATUS']
    X_Var=[i for i in data_final_vars if i not in Y_Var]
    
    y = df_final['POLICY_STATUS']
    X = df_final[X_Var]
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    
    signature = infer_signature(X_test, y_test)
    
    df_Metrics = DataFrame(columns = ['Model','Accuracy','Precision','Recall','F1 Score'] )
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    
    logreg = LogisticRegression()
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    y_pred_logreg=logreg.predict(X_test)
    
    consolidate = ['Logistic Regression',metrics.accuracy_score(y_test, y_pred_logreg),metrics.precision_score(y_test, y_pred_logreg),metrics.recall_score(y_test, y_pred_logreg),metrics.f1_score(y_test, y_pred_logreg)]
    
    df_Metrics.loc[0] = consolidate
    
    #Gaussian Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    nb=GaussianNB()
    
    # fit the model with data
    nb.fit(X_train,y_train)
    y_pred_nb=nb.predict(X_test)
    
    consolidate = ['Gaussian Naive Bayes',metrics.accuracy_score(y_test, y_pred_nb),metrics.precision_score(y_test, y_pred_nb),metrics.recall_score(y_test, y_pred_nb),metrics.f1_score(y_test, y_pred_nb)]
    
    df_Metrics.loc[1] = consolidate
    
    
    
    #Stochastic Gradient Descent
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss = 'modified_huber',shuffle = True,random_state = 101)
    
    # fit the model with data
    sgd.fit(X_train,y_train)
    y_pred_sgd=sgd.predict(X_test)
    
    consolidate = ['Stochastic Gradient Descent',metrics.accuracy_score(y_test, y_pred_sgd),metrics.precision_score(y_test, y_pred_sgd),metrics.recall_score(y_test, y_pred_sgd),metrics.f1_score(y_test, y_pred_sgd)]
    
    df_Metrics.loc[2] = consolidate
    
    # K-nearest Neighbours
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 15)
    
    # fit the model with data
    knn.fit(X_train,y_train)
    y_pred_knn=knn.predict(X_test)
    
    consolidate = ['K-nearest Neighbours',metrics.accuracy_score(y_test, y_pred_knn),metrics.precision_score(y_test, y_pred_knn),metrics.recall_score(y_test, y_pred_knn),metrics.f1_score(y_test, y_pred_knn)]
    
    df_Metrics.loc[3] = consolidate
    
    # Decision Trees
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier(max_depth = 10,random_state = 101)
    
    # fit the model with data
    dtree.fit(X_train,y_train)
    y_pred_dtree=dtree.predict(X_test)
    
    consolidate = ['Decision Trees',metrics.accuracy_score(y_test, y_pred_dtree),metrics.precision_score(y_test, y_pred_dtree),metrics.recall_score(y_test, y_pred_dtree),metrics.f1_score(y_test, y_pred_dtree)]
    
    df_Metrics.loc[4] = consolidate
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfm = RandomForestClassifier(n_estimators = 70,random_state = 101)
    
    # fit the model with data
    rfm.fit(X_train,y_train)
    y_pred_rfm=rfm.predict(X_test)
    
    consolidate = ['Random Forest',metrics.accuracy_score(y_test, y_pred_rfm),metrics.precision_score(y_test, y_pred_rfm),metrics.recall_score(y_test, y_pred_rfm),metrics.f1_score(y_test, y_pred_rfm)]
    
    df_Metrics.loc[5] = consolidate
    
    # Support Vector Machines
    from sklearn.svm import SVC
    svm = SVC(random_state = 101)
    
    # fit the model with data
    svm.fit(X_train,y_train)
    y_pred_svm=svm.predict(X_test)
    
    consolidate = ['Support Vector Machines',metrics.accuracy_score(y_test, y_pred_svm),metrics.precision_score(y_test, y_pred_svm),metrics.recall_score(y_test, y_pred_svm),metrics.f1_score(y_test, y_pred_svm)]
    
    df_Metrics.loc[6] = consolidate
    
    print(df_Metrics)
#     mlflow.log_table('metrics ', df_Metrics)
#     mlflow.log_metrics('metrics', df_Metrics.to_dict(orient='dict'))
    df_Metrics.to_csv('Buying_propensity_Metrices.csv',index=False)
    
    #Model Pickle
    Modell = df_Metrics.loc[df_Metrics["Accuracy"].idxmax()]['Model']

    mlflow.log_param('Model ', str(Modell))
    mlflow.log_metric('Accuracy', df_Metrics.loc[df_Metrics["Accuracy"].idxmax()]['Accuracy'])
    ###########################
    #</save and register model>
    ###########################
    
    registered_model_name= args.registered_model_name
    
    model_dict = {"Logistic Regression":logreg, "Gaussian Naive Bayes": nb, "Stochastic Gradient Descent": sgd , "K-nearest Neighbours": knn,
                  "Decision Trees": dtree, "Random Forest": rfm , "Support Vector Machines": svm}
    
    
#     model_dict = {"Logistic Regression":logreg, "Gaussian Naive Bayes": nb}
    


    # logging the model to the workspace
    print("logging the model via MLFlow")
    model_path = os.path.join(registered_model_name, "trained_model")
    mlflow.pyfunc.log_model(
        model_path, 
#         sk_model= model_dict[Modell],
        python_model=ModelWrapper( model_dict[Modell]),
#         artifacts={"model": model_path},
        signature=signature,
#         registered_model_name= registered_model_name,
#         artifact_path= registered_model_name
    )


    # # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model= model_dict[Modell],
        path=os.path.join(args.model, "trained_model"),
    )
    
    print("Registering ", registered_model_name)
    print("args model: ", args.model)
    print("model_path: ", model_path)
    
    # register logged model using mlflow
    run_id = mlflow.active_run().info.run_id
    print("args run_id: ", run_id)
    model_uri = f'runs:/{run_id}/{model_path}'
    mlflow_model = mlflow.register_model(model_uri, registered_model_name)
    model_version = mlflow_model.version

    # write model info
    print("Writing JSON")
    dict = {"id": "{0}:{1}".format(registered_model_name, model_version)}
    print('dict: ', dict)
#     output_path = os.path.join(model_path, "model_info.json")
#     with open(output_path, "w") as of:
#         json.dump(dict, fp=of)

    mlflow.end_run()

if __name__ == '__main__':
    main()