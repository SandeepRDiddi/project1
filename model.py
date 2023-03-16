import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from arize.pandas.logger import Client, Schema
from arize.utils.types import Environments, ModelTypes

df = pd.read_csv("train_transaction.csv")
df = pd.concat([df[df.isFraud == 0].sample(n=len(df[df.isFraud == 1])), df[df.isFraud == 1]], axis=0)
feature_column_names = ["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "M1", "M2", "M3"]
X = df[feature_column_names]
y = df.isFraud
enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(X)

X = pd.DataFrame(enc.transform(X).toarray(), columns=enc.get_feature_names_out().reshape(-1))
X["TransactionAmt"] = df[["TransactionAmt"]].to_numpy()

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Keep the non one hot encoded DF for monitoring the model
#For example, imagine we had a state variable is_state. 
#With one-hot encoding it would transform into is_state_CA, 
#is_state_NY, is_state_WA, etc. When uploading is_state 
#as a single feature, we can monitor the data of all states.



# Create and train a XGBoost Classifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
model = xgb.fit(X_train, y_train)

# Predict on test set
preds = xgb.predict(X_test)
print(accuracy_score(y_test, preds))

xgb.save_model('xgb_cl_model.json')


#Need to copy original dataframe since X_train is one-hot encoded
orig = df.copy()
train_orig = orig.iloc[X_train.index.values]
train_orig.reset_index(drop=True, inplace=True)

train_preds = xgb.predict(X_train)
train_pred = pd.DataFrame(train_preds, columns=['predictedFraud'])

combined_train_df = pd.concat([train_orig.reset_index(drop=True), train_pred], axis=1)
combined_train_df.fillna('', inplace=True)

#Define the Monitoring Framework 

SPACE_KEY = "3a874f9"
API_KEY = "ba2708632f20ed177c5"

arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)
model_id = (    
   "fraud-detection-tutorial"  # This is the model name that will show up in Arize
)
model_version = "v1.0"  # Version of model - can be any string

#Don't change the values here, we are just making sure you changed the keys to yours
if SPACE_KEY == "YOUR-SPACE-KEY" or API_KEY == "YOUR-API-KEY":    
   raise ValueError("❌ NEED TO CHANGE SPACE AND/OR API_KEY")
else:    
   print("✅ Arize setup complete!")
   
   
training_schema = Schema(    
   prediction_id_column_name="TransactionID",    
   prediction_label_column_name="predictedFraud",    
   actual_label_column_name="isFraud",    
   feature_column_names=feature_column_names,
)


# Logging Training DataFrame
training_response = arize_client.log(    
   dataframe=combined_train_df,    
   model_id=model_id,    
   model_version=model_version,    
   model_type=ModelTypes.SCORE_CATEGORICAL,    
   environment=Environments.TRAINING,    
   schema=training_schema,
)

# If successful, the server will return a status_code of 200
if training_response.status_code != 200:
   print(        
      f"logging failed with response code {training_response.status_code}, {training_response.text}"    
   )
else:    
   print(f"✅ You have successfully logged training set to Arize")


