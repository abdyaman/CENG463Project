import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV



def NDVIcalculate(trainData):
    NIR = trainData["NIR"]
    Red = trainData["Red"]  
    NDVI =  (NIR - Red) / (NIR + Red+0.000001)
    return NDVI

def NDWIcalculate(trainData):
    NIR = trainData["NIR"]
    Green = trainData["Green"]
    NDWI = (Green - NIR) / (Green + NIR+0.000001)
    return NDWI

def decisionTree():
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    method = "Decision Tree"
    showScores(method, y_pred)
    return y_pred

def KNNClassification():   
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 31))}
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1_score')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_k = best_params['n_neighbors']
    knn = KNeighborsClassifier(best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    method = "KNN"
    showScores(method, y_pred)
    print(best_k)
    return y_pred

def gaussianNaiveBayes():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    method = "GNB"
    showScores(method, y_pred)
    return y_pred

def showCorrelation(trainData):
    pearsonCorr = trainData.corr()
    print("\nPearson correlation matrix for 'Code' feature:\n\n",pearsonCorr["Code"])
    sns.heatmap(pearsonCorr, annot=True, cmap='coolwarm')
  
def combineTestData(X_testImported,submission):
    newdf = pd.DataFrame(index = range(0,4620309), columns=["Id","Code", "Blue", "Green", "Red", "NIR"])
    IdTemp = submission.iloc[:,0:1].values
    CodeTemp = submission.iloc[:,1:2].values
    BlueTemp = X_testImported.iloc[:,0:1].values
    GreenTemp = X_testImported.iloc[:,1:2].values
    RedTemp = X_testImported.iloc[:,2:3].values
    NIRTemp = X_testImported.iloc[:,3:4].values
    newdf = newdf.assign(Id=IdTemp, Code=CodeTemp, Blue = BlueTemp, Green = GreenTemp, Red=RedTemp, NIR=NIRTemp)
    # newdf = newdf[newdf.Red != 0]
    # newdf = newdf[newdf.Code != 0]
    
    return newdf

def showScores(method, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(method, " Accuracy: ", accuracy)
    f1 = f1_score(y_test, y_pred, average = "macro")
    print(method, " macro:", f1)
    f1 = f1_score(y_test, y_pred, average = "micro")
    print(method, " micro:", f1)
    f1 = f1_score(y_test, y_pred, average = "weighted")
    print(method," weighted:", f1)
    
def testTrainDataPreparation():
   
    trainData = pd.read_csv("train.csv", low_memory=False)
    X_testImported = pd.read_csv("test.csv", low_memory=False)
    submission = pd.read_csv("submission.csv", low_memory=False)
    
    trainData.drop("Id", axis=1, inplace=True)
    # trainData = trainData[trainData.Red != 0]#Because of the property of dataset, it is enough to drop the "0" values of an input feature feature. All columns are getting cleaned from "0" values. 
    # trainData = trainData[trainData.Code != 0]
    trainData["NDWI"] = NDWIcalculate(trainData)
    trainData["NDVI"] = NDVIcalculate(trainData)
    trainData["Blue"] = trainData["Blue"]
    trainData["Blue"] = trainData["Blue"]/ 1000
    trainData["NIR"] = trainData["NIR"] / 1000
    
    showCorrelation(trainData)
    trainData.drop("Green", axis=1, inplace=True)
    trainData.drop("Red", axis=1, inplace=True)

    X_testImported.drop("Id", axis=1, inplace=True)
    testData = combineTestData(X_testImported,submission)
    testData["NDWI"] = NDWIcalculate(testData)
    testData["NDVI"] = NDVIcalculate(testData)
    testData.drop("Green", axis=1, inplace=True)
    testData.drop("Red", axis=1, inplace=True)
    testData["Blue"] = testData["Blue"] / 1000
    testData["NIR"] = testData["NIR"] / 1000
    idData = testData.iloc[:,0:1].values
    testData.drop("Id", axis=1, inplace=True)
    
    y_train = trainData.iloc[:,0:1].values
    X_train = trainData.iloc[:,1:5].values
    y_test = testData.iloc[:,0:1].values
    X_test = testData.iloc[:,1:5].values
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    X_test = X_test.reshape(-1,4)
    X_train = X_train.reshape(-1,4)
    return X_train, X_test, y_train, y_test, idData

X_train, X_test, y_train, y_test, idData = testTrainDataPreparation()

KNNResult = KNNClassification()
# decisionTreeResult = decisionTree()
# NBresult = gaussianNaiveBayes()

# decisionTreeData = pd.DataFrame(index = range(0,4620309), columns=["Id","Code"])
# decisionTreeData = decisionTreeData.assign(Id=idData, Code=decisionTreeResult)
# KNNData = pd.DataFrame(index = range(0,4620309), columns=["Id","Code"])
# KNNData = KNNData.assign(Id=idData, Code=KNNResult)
# NBdata = pd.DataFrame(index = range(0,4620309), columns=["Id","Code"])
# NBdata = NBdata.assign(Id=idData, Code=NBresult)

# decisionTreeData.to_csv('DTsubmission.csv', index=False)
# KNNData.to_csv('KNNsubmission.csv', index=False)
# NBdata.to_csv('NBsubmission.csv', index=False)