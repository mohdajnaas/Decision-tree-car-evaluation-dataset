import pandas as pd 
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df = pd.read_csv(r"/home/student/Desktop/ajnaas/diabetes.csv")
print(df.head(0))

feature_col = ['Pregnancies','Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = df[feature_col]
y = df.Outcome

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 1 )

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

print(cm)
print(cr)
print(acc)



