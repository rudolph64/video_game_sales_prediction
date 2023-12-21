#load libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


# Load the dataset
video = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
video.info()

#remove the User_Count col as it has too many outliners
video=video.drop(columns=['User_Count'])

video_plat = video[(video['Platform'] == 'PS3') | (video['Platform'] == 'PSP')| (video['Platform'] == 'PS')

| (video['Platform'] == 'XB') | (video['Platform'] == 'PS4') | (video['Platform'] == 'X360') | (video['Platform'] == 'XOne')

| (video['Platform'] == 'PC') | (video['Platform'] == 'Wii') | (video['Platform'] == 'WiiU')]

#now use this dataset
data = video_plat.dropna(subset=['Critic_Score'])

#replace NA with mode and mean
data["User_Score"].fillna((data["User_Score"].median()), inplace=True)
data["Year_of_Release"].fillna((data["Year_of_Release"].median()), inplace=True)
data['Developer'] = data['Developer'].fillna(data['Developer'].mode()[0])
data['Rating'] = data['Rating'].fillna(data['Rating'].mode()[0])
data['Publisher'] = data['Publisher'].fillna(data['Publisher'].mode()[0])

# Identify non-numeric categorical columns
#categorical_columns = video.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#for col in categorical_columns:
    #video[col] = label_encoder.fit_transform(video[col])

data['Name']=le.fit_transform(data['Name'])
data['Platform']=le.fit_transform(data['Platform'])
data['Genre']=le.fit_transform(data['Genre'])
data['Publisher']=le.fit_transform(data['Publisher'])
data['Developer']=le.fit_transform(data['Developer'])
data['Rating']=le.fit_transform(data['Rating'])

selected_features2 = ['Platform','Year_of_Release','Genre','Publisher','Critic_Score','Critic_Count',
                     'User_Score','Rating', 'NA_Sales', 'EU_Sales',
                     'Other_Sales', 'JP_Sales']
X2 = data[selected_features2]
y2 = data['Global_Sales']

#split the data to test & train
X_train,X_test,y_train,y_test=tts(X2,y2,test_size=0.25,random_state=42)

#Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train a Linear Regression model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE) for Linear Regression:', rmse)

# Make pickle file of our model
import pickle
pickle.dump(model, open("model.pkl", "wb"))
print ("pickle complete")

# Predict global sales for a new game (replace with actual values)
# selected rank 2 from actual dataset
#selected_features2 = ['Platform','Year_of_Release','Genre','Publisher','Critic_Score','Critic_Count',
#'User_Score','Rating', 'NA_Sales', 'EU_Sales',
#'Other_Sales', 'JP_Sales']

test_case = pd.DataFrame({
'Platform':[5],
'Year_of_Release':[2006],
'Genre':[10],
'Publisher':[150],
'Critic_Score':[76.0],
'Critic_Count':[51.0],
'User_Score':[8.0],
'Rating':[1],
'NA_Sales': [41.36],
'EU_Sales': [28.96],
'Other_Sales': [8.45],
'JP_Sales': [3.77],
})
predicted_global_sales = model.predict(test_case)
print("Predicted Global Sales",predicted_global_sales[0])
