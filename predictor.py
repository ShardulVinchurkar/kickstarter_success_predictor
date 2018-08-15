import kickstarter as app
import pandas as pd

# Input testing data
input_data = pd.read_csv("test.csv")

# Run the kickstarter app to calculate the predictions
predictions = app.function_predict(input_data)

# Building resultant set
result_set = {"Project Name" : input_data['name'], "Prediction" : predictions}

result = pd.DataFrame(result_set)

# Writing to csv file
result.to_csv("result.csv")

# Printing accuracy of the model on the console
print("Model Accuracy : ", app.model_test(), "%")