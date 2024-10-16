from django.shortcuts import render


# for training data and making predictions
import numpy as np   # for arrays
import pandas as pd  # for datasets
from sklearn.model_selection import train_test_split , cross_val_score # for spliting data and checking accuracy more accurately
from sklearn.ensemble import RandomForestClassifier   # for prediction
from sklearn.metrics import accuracy_score   # for checking accuracy
# Create your views here.

def index(request):
    return render(request, 'index.html')

def submit(request):
    if request.method=="POST":
        pregnancies=request.POST.get('pregnancies')
        glucose=request.POST.get('glucose')
        bp=request.POST.get('bp')
        skin_thickness=request.POST.get('skin_thickness')
        insulin=request.POST.get('insulin')
        dpf=request.POST.get('dpf')
        BMI=request.POST.get('BMI')
        age=request.POST.get('age')

    # training data
    # calling dataset
    df=pd.read_csv("data/DiebetiesDataset.csv")

    cols=df.columns  # columns name 
        
    # for outliers, setting quantile
    lower=0.03       
    upper=0.97
    # setting boundries
    lower_bonds=df[cols].quantile(lower)
    upper_bonds=df[cols].quantile(upper)

    # creating new dataset with less or no outliers by setting range
    df_new=pd.DataFrame()
    df_new[cols]=np.where(df[cols]>=upper_bonds,
                          upper_bonds,
                          np.where(df[cols]<=lower_bonds,
                                   lower_bonds,
                                   df[cols]))
    

    # training dataset
    # input
    x=df_new.drop(columns=[ "Outcome"])
    # output
    y=df_new["Outcome"]

    #spliting data
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

    # training data using RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=110, random_state=42)
    clf.fit(x_train, y_train)
    y_pred= clf.predict(x_test)
    
    # checking accuracy
    accuracy= accuracy_score(y_test, y_pred)
    crossval= np.mean(cross_val_score(clf, x, y, scoring="accuracy", cv=15))

    # predicting user data
    new_data= np.array([[pregnancies, glucose, bp, skin_thickness, insulin, dpf, BMI, age]])
    prediction= clf.predict(new_data)
    if prediction==1:
        output="User has diabeties"
    else:
        output ="User has no diabeties"
    context={
        "preg":pregnancies,
        "glucose": glucose,
        "bp":bp,
        "skin_thickness":skin_thickness,
        "insulin":insulin,
        "dpf": dpf,
        "BMI":BMI,
        "age":age,
        "accuracy":accuracy,
        "cvs": crossval,
        "output": output,
    }
    return render(request, "submit.html", context)