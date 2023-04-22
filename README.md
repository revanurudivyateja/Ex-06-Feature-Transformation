# Ex-06-Feature-Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file

## PROGRAM:
```
Name : R.Divya teja
Register Number : 212220040132
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("/content/Data_to_Transform.csv")
df
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```
## OUTPUT:

![im1](https://user-images.githubusercontent.com/127843136/232397504-d5a65780-bd46-4741-8882-4f431a4aec11.png)

![im2](https://user-images.githubusercontent.com/127843136/232397532-437d158d-c848-4d3f-a4bc-bff603ac801c.png)

![im3](https://user-images.githubusercontent.com/127843136/232397619-fbdb68dc-c462-4862-b395-3bc082cd27ea.png)

![im4](https://user-images.githubusercontent.com/127843136/232397630-2eaa46f1-4bee-4abb-acec-9c2e10a2f9ff.png)

![im5](https://user-images.githubusercontent.com/127843136/232397650-336d883d-6b95-4b08-ae90-ddaae51b6190.png)

![im6](https://user-images.githubusercontent.com/127843136/232397663-db2a37d8-d0e6-4cac-97a0-bb5bd9031e85.png)

![im7](https://user-images.githubusercontent.com/127843136/232397676-52c1e88f-456a-4e0f-9ed3-42a662ec03bf.png)

![im8](https://user-images.githubusercontent.com/127843136/232397714-535d22e9-b837-43a5-8c8f-4050f45dc046.png)

![im9](https://user-images.githubusercontent.com/127843136/232397738-7e685edb-9adc-4f27-a786-efe27979c8f6.png)

![im10](https://user-images.githubusercontent.com/127843136/232397761-72b13f99-e939-4e83-b91d-09baa7982c8b.png)

![im11](https://user-images.githubusercontent.com/127843136/232397777-a52b1bf1-c95f-456f-bb5b-6e98e57cfca5.png)


## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
