import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline mport Pipeline
def main():
  titanic=sns.load_dataset('titanic')
  X=titanic[['pclass','sex','age','sibsp','parch','fare','embarked']]
  y=titanic['survived']
  numeric_features=['age','sibsp','parch','fare']
  numeric_transformer=SimpleImputer(strategy='mean')
  categorical_features=['pclass','sex','embarked']
  categorical_transformer=Pipeline(steps=[
  ('import',SimpleImputer(strategy='most_frequent')),
  ('onehot',
