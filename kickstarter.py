
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Input Data
projects_df = pd.read_csv("data.csv", index_col=None)

# Renaming the labels to failed and successful (Assumed Live and successful to be SUCCESSFUL and rest everything FAILED)
projects_df['is_success'] = projects_df['state'].apply(lambda x : 0 if x == "failed" or x == "canceled" or x == "suspended" or x == "undefined" else 1)

# Preparing the data for train test split
label = projects_df['is_success']
df = projects_df.drop('is_success', 1)

# Stratified splitting of data in 90/10%
training_data, testing_data, training_label, testing_label = train_test_split(df, label, test_size = 0.1, stratify = label)

# Data Pre-processing and transforming
training_data = training_data.drop(['ID', 'name', 'category', 'currency', 'goal', 'pledged', 'state', 'usd pledged'], axis = 1)

# Splitting of start date and time from launched attribute
start_time_split = pd.DataFrame(training_data.launched.str.split(' ').tolist(), columns = ['start','time'])
training_data = pd.concat([training_data.reset_index(drop=True), start_time_split.reset_index(drop=True)], axis=1)

# Adding a new attribute as days instead of start and deadline
training_data['deadline'] = pd.to_datetime(training_data['deadline'])
training_data['start'] = pd.to_datetime(training_data['start'])
set(map(type, training_data.start.values.tolist()))
set(map(type, training_data.deadline.values.tolist()))

training_data['days'] = training_data['deadline'] - training_data['start']

training_data = training_data.drop(['deadline', 'launched', 'start', 'time'], axis = 1)

training_data['days'] = training_data['days'] / np.timedelta64(1, 'D')

# Normalizing the days attribute
training_data['normalized_days'] = stats.boxcox(training_data['days'])[0]

# Normalizing the goal attribute
training_data['normalized_usd_goal_real'] = stats.boxcox(training_data['usd_goal_real'])[0]

# Normalizing the pledged attribute
training_data['usd_pledged_real'] += 1

training_data['normalized_usd_pledged_real'] = stats.boxcox(training_data['usd_pledged_real'])[0]

# Normalizing the backers attribute
training_data['backers'] += 1
training_data['normalized_backers'] = stats.boxcox(training_data['backers'])[0]

# Cleaning unnecessary attributes
training_data = training_data.drop(['backers', 'usd_pledged_real', 'usd_goal_real', 'days'], axis = 1)

# Encoding categorical variables
training_data = pd.get_dummies(training_data, columns = ['main_category', 'country'])

# Training of model
regression_model = linear_model.LinearRegression()
regression_model.fit(training_data, training_label)

# The below function will perform transformations on testing data and bring it to the same vector space as that of
# trained model.
def transform_new_data(new_data):
    new_data = new_data.drop(['ID', 'name', 'category', 'currency', 'goal', 'pledged', 'state', 'usd pledged'], axis=1)

    start_time_split_new_data = pd.DataFrame(new_data.launched.str.split(' ').tolist(), columns=['start', 'time'])
    new_data = pd.concat([new_data.reset_index(drop=True), start_time_split_new_data.reset_index(drop=True)], axis=1)

    new_data['deadline'] = pd.to_datetime(new_data['deadline'], dayfirst=True)
    new_data['start'] = pd.to_datetime(new_data['start'], dayfirst=True)
    set(map(type, new_data.start.values.tolist()))
    set(map(type, new_data.deadline.values.tolist()))

    new_data['days'] = new_data['deadline'] - new_data['start']

    new_data = new_data.drop(['deadline', 'launched', 'start', 'time'], axis=1)

    new_data['days'] = new_data['days'] / np.timedelta64(1, 'D')

    new_data['normalized_days'] = stats.boxcox(new_data['days'])[0]

    new_data['normalized_usd_goal_real'] = stats.boxcox(new_data['usd_goal_real'])[0]

    new_data['usd_pledged_real'] += 1
    new_data['normalized_usd_pledged_real'] = stats.boxcox(new_data['usd_pledged_real'])[0]

    new_data['backers'] += 1
    new_data['normalized_backers'] = stats.boxcox(new_data['backers'])[0]

    new_data = new_data.drop(['backers', 'usd_pledged_real', 'usd_goal_real', 'days'], axis=1)

    new_data = pd.get_dummies(new_data, columns=['main_category', 'country'])

    # Handling the missing attributes after encoding categorical attributes in testing data
    missing_cols = set(training_data.columns) - set(new_data.columns)

    for col in missing_cols:
        new_data[col] = 0

    new_data = new_data[training_data.columns]

    return new_data

# The below function predicts and generates a list of predictions for the new testing data
def function_predict(input_data):
    transformed_new_data = transform_new_data(input_data)
    predictions = regression_model.predict(transformed_new_data)

    # Generating the final predictions
    predictions_list = []
    for n in range(len(predictions)):
        if predictions[n] < 0.4:
            predictions_list.append("Failed")
        else:
            predictions_list.append("Successful")

    return predictions_list

# The below function evaluates the accuracy of the trained model.
def model_test():
    transformed_data = transform_new_data(testing_data)
    test_predictions = regression_model.predict(transformed_data)

    predictions = []
    for index in range(len(test_predictions)):
        if test_predictions[index] < 0.4:
            predictions.append(0)
        else:
            predictions.append(1)

    # Calculating accuracy
    accuracy = accuracy_score(testing_label, predictions) * 100
    return accuracy