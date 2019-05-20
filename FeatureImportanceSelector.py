
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FeatureImportanceSelector:
    """ Class that can slice a CSV file based on feature importance """
    
    def __init__(self, filename, csv_seperator, output_filename, output_csv_seperator, drop_columns, target):
        self.filename = filename
        self.csv_seperator = csv_seperator
        self.output_filename = output_filename
        self.output_csv_seperator = output_csv_seperator
        self.drop_columns = drop_columns
        self.target = target

        # Read, format, and store data. 
        df = pd.read_csv(self.filename, sep=self.csv_seperator)
        df = df.drop(columns=self.drop_columns)
        self.df = df

    def __get_X(self):
        return self.df.drop(self.target, axis=1).copy()
    
    def __get_Y(self):
        return self.df[[self.target]].copy()

    def calculate_accuracy_score(self, num_tests=10, train_size=0.7):
        """ Calculate an estimated prediction classification accuracy given a dataset with a single target """
        score_sum = 0
        for _ in range(num_tests):
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.__get_X(), 
                self.__get_Y()[self.target],
                test_size=1-train_size
            )
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            score_sum += accuracy_score(Y_test, Y_pred)
        score = score_sum / num_tests
        return score

    def get_feature_importance(self):
        """ 
        Plots feature importance using a decision tree classifier 
        
        Return: features, importances
        """
        X, Y = self.__get_X(), self.__get_Y()
        clf = DecisionTreeClassifier()
        clf.fit(X, Y)
        return X.columns.values, clf.feature_importances_

    def plot_feature_importances(self):
        features, importances = self.get_feature_importance()
        df_plot = pd.DataFrame()
        df_plot['Importances'] = importances
        df_plot['Features'] = features
        df_plot = df_plot.sort_values(["Importances"], ascending=False)
        sns.set(style="whitegrid")
        ax = sns.barplot(x='Importances', y='Features', data=df_plot)
        plt.show()

    def feature_selection(self, num_features):
        X, Y = self.__get_X(), self.__get_Y()
        features, importances = self.get_feature_importance()
        importances, features = zip(*(sorted(zip(importances, features), reverse=True)))
        selected_features = features[:num_features]
        df = X[[*selected_features]].copy()
        df[self.target] = Y[self.target]
        # Save new dataset to disk (CSV file). 
        df.to_csv(self.output_filename, index=False, sep=self.output_csv_seperator)
