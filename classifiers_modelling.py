from modelling import *
from sklearn.linear_model import SGDClassifier


class classifiers_model_builder(regressor_model_builder):
        def __init__(self):
            super().__init__()
            
        def load_classify_data(self):
            X, y=self.load_airbnb(self.df,tg_column='Price_Night')
            X = preprocessing.normalize(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.15)
        
            model = LogisticRegression(max_iter=10000)
            model.fit(X_train, y_train)

            y_hat_test=model.predict(X_test)
            y_hat_train=model.predict(X_train)
            y_hat_validation=model.predict(X_validation)


            print("Test: Accuracy:", accuracy_score(y_test, y_hat_test))
            print("Test: Precision:", precision_score(y_test, y_hat_test, average="macro"))
            print("Test: Recall:", recall_score(y_test, y_hat_test, average="macro"))
            print("Test: F1 score:", f1_score(y_test, y_hat_test, average="macro"))

            print("Train: Accuracy:", accuracy_score(y_train, y_hat_train))
            print("Train: Precision:", precision_score(y_train, y_hat_train, average="macro"))
            print("Train: Recall:", recall_score(y_train, y_hat_train, average="macro"))
            print("Train :F1 score:", f1_score(y_train, y_hat_train, average="macro"))

            #return (X_train, X_test, X_validation, y_train, y_test, y_validation)


        def tune_classification_model_hyperparameters(self,model , X_train,X_test,X_validation, y_train,y_test,y_validation):
        #define range of parameters 
            param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1],
                        'Solver': ['lbfgs','sag','saga','newton-cg'],
                        'Maximum iterations': [10, 100, 1000],
                        'Learning rate': ['constant', 'optimal', 'invscaling'],
                        'tol': [0.0001, 0.001, 0.01]}
            
            #instantiate GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5,n_jobs=-1)
            
            #fit model to data
            grid_search.fit(X_train, y_train)

            # with validation set use perfomance metrics to make decison on model

            grid_search.best_estimator_.fit(X_train, y_train)
            y_pred_validation= grid_search.best_estimator_.predict(X_validation)

            acc_score=accuracy_score(y_validation,y_pred_validation)
            performance_metrics=dict(validation_accuracy=acc_score)
            model_name = type(model).__name__
        
            #return best parameters
            self.save_model(grid_search.best_estimator_ , grid_search.best_params_,performance_metrics,'models/classification/'+model_name +'/')

        def evaluate_all_models(self):
            models = [SGDClassifier(early_stopping=True)
            ]

            X_train, X_test, X_validation, y_train, y_test, y_validation = self.load_classify_data()
            for model in models:
                print(type(model))
                model_name = type(model).__name__
                print(model_name)
                best_model,best_params,performance_metrics = self.tune_classification_model_hyperparameters(model,X_train,X_test,X_validation,y_train,y_test,y_validation)
                self.save_model(best_model, best_params, performance_metrics, 'models/classification/'+model_name +'/')
                print(performance_metrics)

        def find_best_model(self,model_evaluator_metric='validation_accuracy'):
            models = [
                SGDClassifier()
            ]
            X_train, X_test, X_validation, y_train, y_test, y_validation = self.load_classify_data()
            best_model = None
            best_params = {}
            best_metrics = {}
            for model in models:
                print(type(model))
                model_name = type(model).__name__
                print(model_name)
                current_model,current_params,performance_metrics = self.tune_classification_model_hyperparameters(model,X_train,X_test,X_validation,y_train,y_test,y_validation)
                if best_model is None or performance_metrics[model_evaluator_metric] < best_metrics[model_evaluator_metric]:
                    best_model = current_model
                    best_params = current_params
                    best_metrics = performance_metrics
            return best_model, best_params, best_metrics
        
Classifier=classifiers_model_builder()
Classifier.load_classify_data()