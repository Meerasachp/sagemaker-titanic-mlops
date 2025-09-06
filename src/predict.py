import sagemaker
import pandas as pd

# Connect to existing endpoint
session = sagemaker.Session()
predictor = sagemaker.predictor.Predictor(
    endpoint_name="titanic-xgboost-endpoint",
    sagemaker_session=session,
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

# Example passenger data (already encoded as numeric)
test_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 1,      # male=1, female=0 (based on preprocessing)
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0
}])

# Convert to CSV row
prediction = predictor.predict(test_passenger.to_csv(header=False, index=False))

print("🔮 Prediction result:", prediction)

