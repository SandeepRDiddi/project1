from arize.pandas.logger import Client, Schema
from arize.utils.types import Environments, ModelTypes

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