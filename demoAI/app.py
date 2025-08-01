import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split#to trian and test and split
from sklearn.linear_model import LogisticRegression#it is for input 
import joblib
df=pd.read_csv('feeds.csv')

label_encoder=LabelEncoder()#by using convert the non num to numerical data
df['action']=label_encoder.fit_transform(df['action'])

X=df[['temperature','humidity']]#using label data spliting the column in x variable
Y=df['action']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)      
joblib.dump(model,'ac-model.pkl')
print("model trained and downloaded successfully")

