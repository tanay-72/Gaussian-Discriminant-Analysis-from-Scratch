import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv('/Users/tanay/Desktop/Cancer/cancerData.csv')

data.drop(columns = ['id' , 'Unnamed: 32'] , inplace = True)
data['diagnosis'] = data['diagnosis'].replace({
        'B':0,
        'M':1
    })

X = data.iloc[:,1:]
Y = data.iloc[:,0]

Xtrain , Xtest , Ytrain , Ytest = train_test_split(X , Y , test_size = 0.2 , random_state=42)
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

class GDA:
    """Gaussian Discriminant Analysis (LDA , shared covariance)"""
    def __init__(self):
        self.mu0 = None
        self.mu1 = None
        self.phi = None 
        self.covariance = None
    def fit(self , Xtrain , Ytrain):
        Xtrain = np.asarray(Xtrain)
        Ytrain = np.asarray(Ytrain)
        
        m = Xtrain.shape[1]
        n = Xtrain.shape[0]
        
        self.mu0 = np.zeros(m)
        self.mu1 = np.zeros(m)
        self.covariance = np.zeros((m,m))
        
        self.phi = np.sum(Ytrain == 1) / n 
            
        for i in range(n):
            if(Ytrain[i] == 0):
                self.mu0 += Xtrain[i]
            else :
                self.mu1 += Xtrain[i]
        self.mu0 /= np.sum(Ytrain == 0)
        self.mu1 /= np.sum(Ytrain == 1) 
        
        for i in range(n):
            if(Ytrain[i] == 0):
                a = Xtrain[i] - self.mu0
                self.covariance += np.outer(a , a.T)
            else :
                b = Xtrain[i] - self.mu1
                self.covariance += np.outer(b , b.T)
        self.covariance /= n 
        self.covariance += 1e-6 * np.eye(m)
        
    def predict(self , Xtest):
        Xtest = np.asarray(Xtest)
        Ypred = []
        coin = np.linalg.inv(self.covariance)
        
        for i in range(Xtest.shape[0]):
            a = Xtest[i] - self.mu0 
            prob_0 = -0.5 * np.dot(a.T ,np.dot(coin, a)) + np.log(1-self.phi)
            b = Xtest[i] - self.mu1 
            prob_1 = -0.5 * np.dot(b.T ,np.dot(coin, b)) + np.log(self.phi)
            if(prob_0 >= prob_1):
                Ypred.append(0)
            else :
                Ypred.append(1)
                
        return Ypred


gla = GDA()
gla.fit(Xtrain , Ytrain)
Ypred = gla.predict(Xtest)
Ypred = np.asarray(Ypred)
Ytest = Ytest.to_numpy()
accuracy = np.mean(Ypred == Ytest)
print("Accuracy:", accuracy)
print(confusion_matrix(Ytest, Ypred))
print(classification_report(Ytest, Ypred))


plt.figure(figsize=(6,4))
sns.histplot(data[data['diagnosis']==0]['radius_mean'],label='Benign', kde=True)
sns.histplot(data[data['diagnosis']==1]['radius_mean'],label='Malignant', kde=True)
plt.legend()
plt.title("Feature Distribution: radius_mean")
plt.show()
