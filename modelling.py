from tabular_data import data_prepearation
import pandas as pd 
from sklearn import datasets, model_selection
from aicore.ml import data
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tabular_data import data_prepearation
import pandas as pd 
from sklearn import datasets, model_selection
from aicore.ml import data
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV   
from sklearn.model_selection import GridSearchCV 

class model_builder(data_prepearation):
    def __init__(self):
        super().__init__()

       



    def load_data(self):
        np.random.seed(2)
        self.load_airbnb(self.df,'Price_Night')
        # self.df['ID'] = pd.to_numeric(self.df['ID'])
        # Use `data.split` to split the data into training, validation, and test sets.

        X, y=self.load_airbnb(self.df,tg_column='Price_Night')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_test, y_test, test_size=0.3
        )
        # create SGD regressor object

        scaler = StandardScaler()
        # #print(X_test.shape, X_train.shape)
        # fit the scaler on the training data
        scaler.fit(X_train)


        #transform the training and test data
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_validation = scaler.transform(X_validation)

        # X_train = preprocessing.normalize(X_train)
        # X_test = preprocessing.normalize(X_test)
        # X_validation = preprocessing.normalize(X_validation)

    
        model = SGDRegressor()

        # fit the model
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        y_pred_validation= model.predict(X_validation)

        train_loss = mean_squared_error(y_train, y_pred_train)
        test_loss = mean_squared_error(y_test, y_pred)
        validation_loss = mean_squared_error(y_validation, y_pred_validation)
        rmse_validation_score = np.sqrt(validation_loss)

        mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=2)
        r_squared = mse / var
        # summarize results
        print('MSE from bias_variance lib [avg expected loss]: %.3f' % mse)
        print('Avg Bias: %.3f' % bias)
        print('Avg Variance: %.3f' % var)
        print('Mean Square error by Sckit-learn lib: %.3f' % metrics.mean_squared_error(y_test,y_pred))
        print('R_squared: %.3f' % r_squared) 




       

    # To tune hyper parameters using random search 

    def custom_tune_regression_model_hyperparameters(model, X_train,X_test,X_validation, y_train,y_test,y_validation,params):


        # Create the GridSearchCV object to search over the parameter space
        grid_search = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # with validation set use perfomance metrics to make decison on model

        grid_search.best_estimator_.fit(X_train, y_train)
        y_pred_validation= grid_search.best_estimator_.predict(X_validation)


        validation_loss = mean_squared_error(y_validation, y_pred_validation)
        rmse_validation_score = np.sqrt(validation_loss)

        performance_metrics=dict(RMSE_validation=rmse_validation_score)
       
        

    # Return the best parameters and model
        print('Best grid search hyperparameters are: '+str(grid_search.best_params_))
        print('Best estimator is:' +str(grid_search.best_estimator_))
        print('Perfomance metrics on validation set:' +str(performance_metrics))
        return (grid_search.best_params_,grid_search.best_estimator_,performance_metrics)



    def tune_hyper_par(self):
        np.random.seed(2)
        self.load_airbnb(self.df,'Price_Night')
        # Use `data.split` to split the data into training, validation, and test sets.

        X, y=self.load_airbnb(self.df,tg_column='Price_Night')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)

        scaler = StandardScaler()
        # #print(X_test.shape, X_train.shape)
        # fit the scaler on the training data
        scaler.fit(X_train)


        #transform the training and test data
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_validation = scaler.transform(X_validation)

        model = SGDRegressor(max_iter=1000)

        learning_rate= ['constant','optimal','invscaling','adaptive']
        loss= ['squared_error','huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        penalty=['l2', 'l1', 'elasticnet', None]
        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]


        # Define hyperparameter distributions

        param_distributions = dict(loss=loss,
                            penalty=penalty,
                            alpha=alpha,
                            learning_rate=learning_rate)

        # Create a RandomizedSearchCV object
 
 

        random_search_cv = RandomizedSearchCV( model, param_distributions=param_distributions, cv=3, n_iter=512, n_jobs=-1)
       

        random_search_cv.fit(X_train, y_train)

        print('Best random search hyperparameters are: '+str(random_search_cv.best_params_))
        print('Best random search score is: '+str(random_search_cv.best_score_))








    

model_class=model_builder()

model_class.load_data()
#model_class.tune_hyper_par()



                
                