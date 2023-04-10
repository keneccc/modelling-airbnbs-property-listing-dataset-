# Milestone 1  

Cleaned csv tabular data table 
Converted CSV data to pandas data framed. Cleaned Tabular data by removing all Nan values from all relevant clomuns 


# Milestone 2 
Clenaed images data 
Using os to access the images folder then resizing every image and saved new edited images in new processed images folder. 

# Milestone 3 

Cleaned csv dtabular data that has been converted to pandas data frame was split into feattures and labels. It was also further cleaned with the dataframe containing only numerical data.

Features and Labels are split into training, test and validation sets of data. All features where then normalised using Scaler from sklearn preprocessing.

The following model's hyperparamenters where tuned with the root mean squared error perfomance metric used to evalute the best models(Models, hyperparameters and perfomance metrics were saved in json and joblib files):
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor & SGDregressor.

All models could be further fine tuned with better hyperparameters and or stacking and blending to improve accuracy. 
Based on the perfomance metrics the Gradient Boosting Regressor was used for predictions

Created Classifier model and set early stopping as true in order to prevent overfitting, model hyperparmeters wer also fine tuned after being evaluated using accuracy
# Milestone 4
Developed neural network regressor model and fine tuned model using learning rate, layer width of neural network, model depth and the chosen optimiser. 
Training runs were monitored and observed using Tensorboard. 

# Conclusions
Developing machine learning models to predict nightly prices. Cleaned and preprocessed both tabular and image data, and have split your data into training, testing, and validation sets. Fine-tuned multiple regression models and a neural network regressor model to achieve the best performance possible.
In order to measure the accuracy of models, used the root mean squared error (RMSE) performance metric and have saved the best models along with their hyperparameters in json and joblib files. Used Tensorboard to monitor and observe the training runs of neural network model.
 
 # Future Enhancements 
 Further hyperparameter tuning, stacking and blending of models, or exploring different architectures for neural network. 
 It's also important to consider the potential impact of predictions and ensure that they are used ethically and responsibly.
 
 
