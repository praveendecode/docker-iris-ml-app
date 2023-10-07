import pandas as pd
from sklearn.linear_model import LogisticRegression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


df = pd.read_csv("Iris_Data.csv")

# Split data 


x = df.drop('species',axis=1)

y = df['species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)


# model creation 

model = LogisticRegression().fit(x_train,y_train)

# Model Evaluation

test = list(map(float,input("Enter values to test the model : ").split()))
prediction = model.predict([test])

# Model Predicted 
y_pred = model.predict(x_test)

# Accuray of the model 

accuracy = accuracy_score(y_test,y_pred)

# Final Outcome

if prediction[0] == 0:
    x = accuracy*100
    print(f'Provided values :{test} ..... Predicted : "Iris-setosa" with the {round(x,2)}% accuracy')
    print()
    print()
elif prediction[0] == 1:
    x = accuracy*100
    print(f'Provided values :{test} ..... Predicted : "Iris-versicolor"  with the {round(x,2)}% accuracy')
    print()
    print()
elif prediction[0] == 2:
    x = accuracy*100
    print(f'Provided values :{test} ..... Predicted : "Iris-virginica"  with the {round(x,2)}% accuracy')
    print()
    print()