#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings

# Disable all warnings
warnings.filterwarnings("ignore")


# In[3]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance

data = pd.read_csv('On_Time_Reporting_1987_2023.csv')


# In[4]:


# find all cols that have all the values as NaN
drop_cols = data.columns[data.isna().all()].tolist()

print('cols to be dropped: start')
print(drop_cols)
print('cols to be dropped: end')

clean_data = data.drop(columns=drop_cols)

print(clean_data.describe())

# Hypothesis Notes
'''
Here our goal is to develop a classification model to predict whether a flight will arrive on time 
(ArrDel15 = 0) or delayed (ArrDel15 = 1). Here (ArrDel15 = 0), (ArrDel15 = 1) are the attributes in dataset 
which will be used to identify whether it is delayed or arrived on time. Also attributes like scheduled and 
actual departure time, arrival time, delays, airtime, distance, and more will help.
The target variable: ArrDel15 (indicating an arrival delay of 15 minutes or more as (1) or if it is not; it is (0).
So, the predictor variables could be CRSDepTime, DepDelay, AirTime, Distance, OriginAirportID, 
DestAirportID, CarrierDelay, WeatherDelay, NASDelay and other delay types.
'''


# # EDA

# In[19]:


# Display basic statistics of the numerical columns
clean_data.describe()


# In[ ]:


import pandas as pd

# Assuming your DataFrame is named 'clean_data'

# Display the number of missing values in each column
missing_values = clean_data.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Option 1: Remove rows with missing values
clean_data_no_missing = clean_data.dropna()

# Option 2: Fill missing values with mean, median, or a specific value
clean_data_filled = clean_data.fillna(clean_data.mean())  # Replace with mean value

# Option 3: Interpolate missing values
clean_data_interpolated = clean_data.interpolate()


# In[ ]:


# Display the number of missing values after handling
missing_values_after = clean_data_interpolated.isnull().sum()
print("\nMissing values per column after handling:")
print(missing_values_after)


# In[ ]:





# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of departure delays
plt.figure(figsize=(12, 6))
sns.histplot(clean_data['DepDelayMinutes'], bins=30, kde=True)
plt.title('Distribution of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.show()


# In[22]:


# Count of cancellations
cancel_counts = clean_data['Cancelled'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
cancel_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Flight Cancellations')
plt.xlabel('Cancellation Status')
plt.ylabel('Number of Flights')
plt.xticks([0, 1], ['Not Cancelled', 'Cancelled'], rotation=0)
plt.show()


# In[26]:


# Distribution of departure delays
sns.histplot(clean_data['DepDelay'], bins=30, kde=True)
plt.title('Distribution of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.show()

# Scatter plot of departure delay vs arrival delay
sns.scatterplot(x='DepDelay', y='ArrDelay', data=clean_data, alpha=0.5)
plt.title('Departure Delay vs Arrival Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.show()


# In[27]:


# Count of cancellations
sns.countplot(x='Cancelled', data=clean_data)
plt.title('Count of Cancelled Flights')
plt.xlabel('Cancelled')
plt.ylabel('Count')
plt.show()

# Count of diversions
sns.countplot(x='Diverted', data=clean_data)
plt.title('Count of Diverted Flights')
plt.xlabel('Diverted')
plt.ylabel('Count')
plt.show()


# In[24]:


# Convert FlightDate to datetime
clean_data['FlightDate'] = pd.to_datetime(clean_data['FlightDate'])

# Plotting number of flights over time
plt.figure(figsize=(14, 6))
clean_data.groupby('FlightDate').size().plot()
plt.title('Number of Flights Over Time')
plt.xlabel('Flight Date')
plt.ylabel('Number of Flights')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


clean_data.columns


# In[6]:


'''
Hypothesis 1: Flights with longer scheduled airtime will have higher likelihood of arrival delays 
due to greater chance of operational issues.Here, Predictor variable can be: CRSElapsedTime, Distance
'''
print("sorted data after dropping cols with all NAs")
print(sorted(list(clean_data)))
# dropping rows with NA / None values
clean_data = clean_data.dropna(subset=['ArrDel15'])
print(clean_data.describe())
'''
['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 
'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline', 'Tail_Number', 'Flight_Number_Reporting_Airline',
 'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 
 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportID', 'DestAirportSeqID', 
 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName', 
 'DestWac', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups', 
 'DepTimeBlk', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay', 
 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'Cancelled', 'CancellationCode',
  'Diverted', 'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Flights', 'Distance', 'DistanceGroup', 
  'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'FirstDepTime', 
  'TotalAddGTime', 'LongestAddGTime', 'DivAirportLandings', 'DivReachedDest', 'DivActualElapsedTime', 
  'DivArrDelay', 'DivDistance', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', 'Div1WheelsOn', 
  'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 
  'Div2AirportSeqID', 'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum', 
  'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', 'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime']

'''

print('\n****** Analyzing Hypothesis #1 ****** ')

print(sorted(list(clean_data)))

list_of_cols = [
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
    'DOT_ID_Reporting_Airline', 'Flight_Number_Reporting_Airline',
    'OriginAirportID', 'OriginCityMarketID',
    'OriginStateFips', 'OriginWac', 'DestAirportID', 'DestAirportSeqID',
    'DestCityMarketID', 'DestStateFips', 'DestWac', 'CRSDepTime',
    'DepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15',
    'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn',
    'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay',
    'ArrDelayMinutes', 'ArrDel15',  'ArrivalDelayGroups',
    'Diverted', 'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Flights',
    'Distance', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay',
    'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'FirstDepTime'
]
columns_with_mean_gt_0 = clean_data[[*list_of_cols]]

print(columns_with_mean_gt_0.corrwith(columns_with_mean_gt_0['ArrDel15']))
# Added new features on checking correlation with ArrDel15
# DepDelay                           0.497804
# DepDelayMinutes                    0.481266
# DepDel15                           0.740944
# DepartureDelayGroups               0.680244
# ArrDelay                           0.562191
# ArrDelayMinutes                    0.512619
# ArrDel15                           1.000000
# ArrivalDelayGroups                 0.756383
# TaxiOut                            0.202383
# WheelsOff                          0.203919
# WheelsOn                           0.089645
clean_data_hypo1 = clean_data[
    ['CRSElapsedTime', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'ArrDel15']]
# print('describe clean_data_hypo1')
# print(clean_data_hypo1.describe())

print(clean_data_hypo1.corr())
X_h1 = clean_data_hypo1[
    ['CRSElapsedTime', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn']]
y_h1 = clean_data_hypo1['ArrDel15']
X_h1_train, X_h1_test, y_h1_train, y_h1_test = train_test_split(X_h1, y_h1, test_size=0.3, random_state=44)
rf_model = RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=44)
rf_model.fit(X_h1_train, y_h1_train)
predictions = rf_model.predict(X_h1_test)

# View accuracy score
rf_score = accuracy_score(y_h1_test, predictions)
print(f'Randon Forest Accuracy Score:{rf_score}')
h1_precision = precision_score(y_h1_test, predictions)
h1_recall = recall_score(y_h1_test, predictions)
print(f'For h1 - Randon Forest : precision score is: {h1_precision} and recall score is: {h1_recall}')


# In[ ]:





# In[28]:


import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
feature_names = X_h1.columns

# Sort the features by importance
indices = importances.argsort()

# Plotting the bar chart
plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()


# In[8]:


'''
Hypothesis 2: Flights departing early in the day have a lower delay risk due to less propagation of delays.
Predictor Variable: CRSDepTime

Using Logistic Regression

'''
print('\n****** Analyzing Hypothesis #2 ****** ')

# X_h1 = clean_data_hypo1[['CRSElapsedTime', 'Distance', 'DepDel15', 'DepartureDelayGroups','TaxiOut','WheelsOff', 'WheelsOn']]
clean_data_hypo2 = clean_data[
    ['CRSDepTime', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'ArrDel15']]
# print('describe clean_data_hypo2')
# print(clean_data_hypo2.describe())
print(clean_data_hypo2.corr())


# In[9]:


# all parameters not specified are set to their defaults
# explicity set max_iter = 1000 as default is 100 and running into
# ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT
logisticRegression = LogisticRegression(max_iter=1000)
X_h2 = clean_data_hypo2[
    ['CRSDepTime', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn']]
y_h2 = clean_data_hypo2['ArrDel15']

X_h2_train, X_h2_test, y_h2_train, y_h2_test = train_test_split(X_h2, y_h2, test_size=0.3, random_state=44)
logisticRegression.fit(X_h2_train, y_h2_train)
lr_predictions = logisticRegression.predict(X_h2_test)

# Use score method to get accuracy of model
score_h2 = accuracy_score(y_h2_test, lr_predictions)
# score_h2 = logisticRegression.score(X_h2_test, y_h2_test)


model_fi = permutation_importance(logisticRegression, X_h2, y_h2)

for ind, col in enumerate(X_h2.columns):
    print(f'The importance of {col} is {round(model_fi.importances_mean[ind] * 100, 2)}%')

print(f'For h2 - Logistic Regression : Accuracy score is: {score_h2}')
h2_precision = precision_score(y_h2_test, lr_predictions, zero_division=1)
h2_recall = recall_score(y_h2_test, lr_predictions)
print(f'For h2 - Logistic Regression  : precision score is: {h2_precision} and recall score is: {h2_recall}')


# In[10]:


cm = metrics.confusion_matrix(y_h2_test, lr_predictions, labels=[0, 1])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap=plt.cm.Oranges);
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.ylabel('Actual label ');
plt.xlabel('Predicted label ');
all_sample_title = 'Accuracy Score: {0}'.format(round(score_h2, 4))
plt.title(all_sample_title, size=15)
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import numpy as np


# Get the confusion matrix
cm = metrics.confusion_matrix(y_h2_test, lr_predictions, labels=[0, 1])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
# Plot bars for each class
bar1 = ax.bar(np.arange(len(class_names)), cm[:, 0], bar_width, label='Class 0', alpha=0.8)
bar2 = ax.bar(np.arange(len(class_names)) + bar_width, cm[:, 1], bar_width, label='Class 1', alpha=0.8)
# Add labels, title, and legend
ax.set_xlabel('True label')
ax.set_ylabel('Normalized Counts')
ax.set_title('Confusion Matrix')
ax.set_xticks(np.arange(len(class_names)) + bar_width / 2)
ax.set_xticklabels(class_names)
ax.legend()
# Display the values on top of the bars
for bar in [bar1, bar2]:
    for rect in bar:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom')

plt.show()


# In[ ]:





# In[11]:


'''
Hypothesis 3: Flights to hub airports have a higher delay risk due to congestion.
Predictors: DestAirportID
'''

print('\n****** Analyzing Hypothesis #3 ****** ')
clean_data_hypo3 = clean_data[
    ['DestAirportID', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'ArrDel15']]
# print(clean_data_hypo3.describe())
print(clean_data_hypo3.corr())

X_h3 = clean_data_hypo3[
    ['DestAirportID', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn']]
y_h3 = clean_data_hypo3['ArrDel15']

X_h3_train, X_h3_test, y_h3_train, y_h3_test = train_test_split(X_h3, y_h3, test_size=0.3, random_state=44)

# Train a model using the scikit-learn API
xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1,
                                   max_depth=3)
xgb_classifier.fit(X_h3_train, y_h3_train)
scores_h3 = cross_val_score(xgb_classifier, X_h3_train, y_h3_train, cv=5)
print("Mean cross-validation score: %.2f" % scores_h3.mean())
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgb_classifier, X_h3_train, y_h3_train, cv=kfold)
print('K-fold CV average score: %.2f' % kf_cv_scores.mean())
xgboost_pred = xgb_classifier.predict(X_h3_test)
max_xgboost_score = max(scores_h3.mean(), kf_cv_scores.mean())
print(f'For h3 - XBoost : Accuracy score is: {max_xgboost_score}')
h3_precision = precision_score(y_h3_test, xgboost_pred, zero_division=1)
h3_recall = recall_score(y_h3_test, xgboost_pred)
print(f'For h3 - XBoost : precision score is: {h3_precision} and recall score is: {h3_recall}')


# In[31]:


# Convert the model to a native API model
model = xgb_classifier.get_booster()
cm_xgboost = confusion_matrix(y_h3_test, xgboost_pred, labels=[0, 1])
cm_xgboost = cm_xgboost.astype('float') / cm_xgboost.sum(axis=1)[:, np.newaxis]

xgb_fea_imp = pd.DataFrame(list(xgb_classifier.get_booster().get_fscore().items()),
                           columns=['feature', 'importance']).sort_values('importance', ascending=False)
print('', xgb_fea_imp)



# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 7))
sns.set(font_scale=1.4)
bottom = [0, 0]
class_labels = ['OnTime', 'Delayed']

for i in range(len(class_labels)):
    plt.bar(
        x=[0, 1], 
        height=cm_xgboost[:, i], 
        bottom=bottom, 
        label=f'Class {i} ({class_labels[i]})', 
        alpha=0.7
    )
    bottom += cm_xgboost[:, i]

# Add labels and legend
plt.xlabel('Predicted Label')
plt.ylabel('Proportion')
plt.title('Confusion Matrix Proportions')
plt.xticks([0, 1], ['OnTime', 'Delayed'])
plt.legend()

plt.show()


# In[ ]:





# In[13]:


'''
Hypothesis 4: Flights with longer taxi-out times or NASDelay have a higher likelihood of arrival delays due to traffic congestion or issue at the origin airport.
Predictor Variable: TaxiOut, NASDelay
'''

print('\n****** Analyzing Hypothesis #4 ****** ')
# NOTE: We are dropping this feature: NASDelay
# dropping NASDelay as the non-null records result with ArrDel15 with 1 as recoreds
clean_data_hypo4 = clean_data[
    ['TaxiOut', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'WheelsOff', 'WheelsOn', 'ArrDel15']]
# print(clean_data_hypo4.describe())
print(clean_data_hypo4.corr())

X_h4 = clean_data_hypo4[['TaxiOut', 'Distance', 'DepDel15', 'DepartureDelayGroups', 'WheelsOff', 'WheelsOn']]
y_h4 = clean_data_hypo4['ArrDel15']
X_h4_train, X_h4_test, y_h4_train, y_h4_test = train_test_split(X_h4, y_h4, test_size=0.3, random_state=44)

# all parameters not specified are set to their defaults using Logistic Regression
logisticRegression_h4 = LogisticRegression()
logisticRegression_h4.fit(X_h4_train, y_h4_train)
lr_predictions_h4 = logisticRegression_h4.predict(X_h4_test)
# Use score method to get accuracy of model
score_h4 = accuracy_score(y_h4_test, lr_predictions_h4)

print(f'For h4 - Regression : Accuracy score is: {score_h4}')
h4_precision = precision_score(y_h4_test, lr_predictions_h4)
h4_recall = recall_score(y_h4_test, lr_predictions_h4)
print(f'For h4 - Regression : precision score is: {h4_precision} and recall score is: {h4_recall}')


# In[ ]:





# In[14]:


model_fi_h4 = permutation_importance(logisticRegression_h4, X_h4, y_h4)

for ind, col in enumerate(X_h4.columns):
    print(f'The importance of {col} is {round(model_fi_h4.importances_mean[ind] * 100, 2)}%')

cm = metrics.confusion_matrix(y_h4_test, lr_predictions_h4, labels=[0, 1])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap=plt.cm.Oranges);
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.ylabel('Actual label ');
plt.xlabel('Predicted label ');
all_sample_title = 'Accuracy Score: {0}'.format(round(score_h4, 4))
plt.title(all_sample_title, size=15)
plt.show()

# Verify the accuracy and the feature importance of all the features and all models
# point to DepartureDelayGroups as the primary feature


# In[32]:


from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_h4_test, lr_predictions_h4)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[33]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_h4_test, lr_predictions_h4)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[34]:


plt.figure(figsize=(12, 6))
sns.histplot(lr_predictions_h4, bins=50, kde=True)
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Class Probability Distribution')
plt.show()


# In[35]:


from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_h4_test, lr_predictions_h4, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot(prob_pred, prob_true, marker='.')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.show()

