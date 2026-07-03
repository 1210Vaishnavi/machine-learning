from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def main():
    iris=load_iris()
    x=iris.data
    y=iris.target
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=42
    )
    model=GaussianNB()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(f"accuracy: {accuracy_score(y_test,y_pred):.2f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test,y_pred))
    print("Classification Report:")
    print(classification_report(y_test,y_pred,target_names=iris.target_names))
if __name__=="__main__":
    main()
accuracy: 1.00
Confusion matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
