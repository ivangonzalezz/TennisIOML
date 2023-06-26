# Tennis stroke classification with ML

This project details how to classify a tennis player stroke with Machine Learning with data gathered from an Apple Watch.

The application required to gather this data is [link text](https://). It needs the following REST API to export data from the device: [link text](https://).

It's recommended to run this notebook at [Google Colab](https://colab.research.google.com), where it's built.


```python
#@title
# Import dependencies
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import matplotlib.ticker as plticker

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

```

## Loading data

Data is loaded from JSON files. Each of them is a list of sensor data from the device at a given timestamp.


```python
# Check files to load
folder = 'trainments'
files = os.listdir(folder)

# Load data from .json files
frames = []
for f in files:
    if '.json' in f:
        d = pd.read_json(f'{folder}/{f}')
        frames.append(d)
        print(f'Loaded {f}')

data = pd.concat(frames, ignore_index=True)
```

    Loaded 20230625_011216.json
    Loaded 20230625_095026.json
    Loaded 20230625_013636.json
    Loaded 20230625_011333.json
    Loaded 20230625_094944.json
    Loaded 20230625_124903.json
    Loaded 20230625_094722.json
    Loaded 20230625_094808.json
    Loaded 20230625_125655.json
    Loaded 20230625_013444.json
    Loaded 20230625_011052.json
    Loaded 20230624_113411.json
    Loaded 20230625_013541.json
    Loaded 20230625_094833.json
    Loaded 20230625_013330.json
    Loaded 20230625_094910.json
    Loaded 20230625_094747.json


## Prepare data

Once loaded, we need to solve some tricky aspects present in data:

* The timestamp gathered from the device has a precision of a second while it generates almost 50 samples for each second. We should expand the precision adding made up milliseconds to each sample.

* Each file contains one kind of movement repeated in a period of time. We need to window the data for each movement done.

* In order to train the model, we should convert all movements in summaries for each period. This will let us use more simple algorithms.

* Finally, we'll split the data into train and test datasets in order to validate the model created with data not used to create it.

### Timestamp precision

First, let's solve the timestamp precision issue. We'll need to know how many samples exist for each second in order to split the second in this number of samples.


```python
# Group samples by second
grouped_per_second = data.groupby(["identifier", "kind", "timestamp"], as_index=False)['identifier'].count()
grouped_per_second['first_index'] = grouped_per_second.apply(lambda row: data[data['timestamp'] == row['timestamp']].index[0], axis=1)

grouped_per_second.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kind</th>
      <th>timestamp</th>
      <th>identifier</th>
      <th>first_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Drive</td>
      <td>2023-06-24 11:34:11</td>
      <td>12</td>
      <td>17738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drive</td>
      <td>2023-06-24 11:34:12</td>
      <td>50</td>
      <td>17750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drive</td>
      <td>2023-06-24 11:34:13</td>
      <td>50</td>
      <td>17800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Drive</td>
      <td>2023-06-24 11:34:14</td>
      <td>50</td>
      <td>17850</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Drive</td>
      <td>2023-06-24 11:34:15</td>
      <td>51</td>
      <td>17900</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Give timestamp millisecond precision and add an attribute to know the time since the trainment start
data['timestamp_millis'] = data['timestamp'].dt.round('L')
data['timestamp_millis'] = data.apply(
    lambda row: row['timestamp_millis'] +
        pd.to_timedelta(
            np.linspace(50, 950, num=grouped_per_second[grouped_per_second['timestamp'] == row['timestamp']].iloc[0]['identifier'])
            [row.name-grouped_per_second[grouped_per_second['timestamp'] == row['timestamp']].iloc[0]['first_index']]
            .astype(int),
            unit='ms'),
    axis=1)

data['time_since_start'] = data['timestamp_millis'] - data['identifier'].astype('datetime64[ns]')
```

    /var/folders/3_/kkbh_njd6r3794xrzcrfrkj00000gp/T/ipykernel_54602/471430095.py:12: UserWarning: Parsing dates in %d-%m-%Y %H:%M:%S format when dayfirst=False (the default) was specified. Pass `dayfirst=True` or specify a format to silence this warning.
      data['time_since_start'] = data['timestamp_millis'] - data['identifier'].astype('datetime64[ns]')



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>kind</th>
      <th>pitch</th>
      <th>roll</th>
      <th>yaw</th>
      <th>timestamp</th>
      <th>xAcceleration</th>
      <th>yAcceleration</th>
      <th>zAcceleration</th>
      <th>xRotation</th>
      <th>yRotation</th>
      <th>zRotation</th>
      <th>timestamp_millis</th>
      <th>time_since_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25-06-2023 01:12:16</td>
      <td>Drive</td>
      <td>-1.052237</td>
      <td>0.473681</td>
      <td>0.278520</td>
      <td>2023-06-25 01:12:16</td>
      <td>-0.024398</td>
      <td>0.001278</td>
      <td>0.001278</td>
      <td>-0.007708</td>
      <td>0.014792</td>
      <td>-0.095422</td>
      <td>2023-06-25 01:12:16.050</td>
      <td>0 days 00:00:00.050000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25-06-2023 01:12:16</td>
      <td>Drive</td>
      <td>-1.053003</td>
      <td>0.471329</td>
      <td>0.276144</td>
      <td>2023-06-25 01:12:16</td>
      <td>0.025908</td>
      <td>-0.014818</td>
      <td>-0.014818</td>
      <td>-0.016041</td>
      <td>-0.029510</td>
      <td>-0.036892</td>
      <td>2023-06-25 01:12:16.150</td>
      <td>0 days 00:00:00.150000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25-06-2023 01:12:16</td>
      <td>Drive</td>
      <td>-1.053979</td>
      <td>0.470275</td>
      <td>0.275632</td>
      <td>2023-06-25 01:12:16</td>
      <td>0.026299</td>
      <td>0.011936</td>
      <td>0.011936</td>
      <td>-0.062542</td>
      <td>-0.011987</td>
      <td>-0.024491</td>
      <td>2023-06-25 01:12:16.250</td>
      <td>0 days 00:00:00.250000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25-06-2023 01:12:16</td>
      <td>Drive</td>
      <td>-1.055339</td>
      <td>0.470844</td>
      <td>0.276280</td>
      <td>2023-06-25 01:12:16</td>
      <td>0.000859</td>
      <td>0.009694</td>
      <td>0.009694</td>
      <td>-0.068776</td>
      <td>-0.006747</td>
      <td>-0.026771</td>
      <td>2023-06-25 01:12:16.350</td>
      <td>0 days 00:00:00.350000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25-06-2023 01:12:16</td>
      <td>Drive</td>
      <td>-1.056421</td>
      <td>0.471384</td>
      <td>0.277111</td>
      <td>2023-06-25 01:12:16</td>
      <td>0.019985</td>
      <td>0.011526</td>
      <td>0.011526</td>
      <td>-0.041852</td>
      <td>-0.005576</td>
      <td>0.012301</td>
      <td>2023-06-25 01:12:16.450</td>
      <td>0 days 00:00:00.450000</td>
    </tr>
  </tbody>
</table>
</div>



### Window data

We're going to window the data manually. First, let's define a filter for values of xAcceleration near 0. This will help us to identify each movement.


```python
# Set global attributes
data['x_accel_smoothed'] = data.apply(lambda row: 0 if row['xAcceleration'] < 0.1 and row['xAcceleration'] > -0.1 else row['xAcceleration'], axis=1)
```

Then, we plot this attribute and we can see a pattern for each movement.


```python
loc = plticker.MultipleLocator(base=1000000000)
sns.set(rc={'figure.figsize':(20,10)})

_tmp = data[data['identifier'] == '25-06-2023 09:50:26']

axes = sns.lineplot(y='x_accel_smoothed', x='time_since_start', data=_tmp)
axes.xaxis.set_major_locator(loc)
```


    
![png](README_files/README_17_0.png)
    


We define the start and the end of each movement in a file called windows.xlsx, that will be used to split the data.


```python
windows = pd.read_excel('{}/windows.xlsx'.format(folder))
windows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Identifier</th>
      <th>From</th>
      <th>To</th>
      <th>Kind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-24 11:34:11</td>
      <td>0.60</td>
      <td>0.90</td>
      <td>Drive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-24 11:34:11</td>
      <td>0.90</td>
      <td>1.20</td>
      <td>Drive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-24 11:34:11</td>
      <td>1.20</td>
      <td>1.85</td>
      <td>Drive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-24 11:34:11</td>
      <td>1.85</td>
      <td>2.15</td>
      <td>Drive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-24 11:34:11</td>
      <td>2.15</td>
      <td>2.45</td>
      <td>Drive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>188</th>
      <td>2023-06-25 09:50:26</td>
      <td>1.05</td>
      <td>1.30</td>
      <td>Backhand</td>
    </tr>
    <tr>
      <th>189</th>
      <td>2023-06-25 09:50:26</td>
      <td>1.30</td>
      <td>1.50</td>
      <td>Backhand</td>
    </tr>
    <tr>
      <th>190</th>
      <td>2023-06-25 09:50:26</td>
      <td>1.50</td>
      <td>1.80</td>
      <td>Backhand</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2023-06-25 09:50:26</td>
      <td>1.80</td>
      <td>2.05</td>
      <td>Backhand</td>
    </tr>
    <tr>
      <th>192</th>
      <td>2023-06-25 09:50:26</td>
      <td>2.05</td>
      <td>2.30</td>
      <td>Backhand</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 4 columns</p>
</div>




```python
splitted_data = []
for index, row in windows.iterrows():
    d = data[
        (data['identifier'] == row['Identifier'].strftime('%d-%m-%Y %H:%M:%S')) &
        (data['time_since_start'] >= pd.to_timedelta(row['From']*10, unit='S')) &
        (data['time_since_start'] < pd.to_timedelta(row['To']*10, unit='S'))
        ]
    splitted_data.append(d)

print(f'Total movements: {len(splitted_data)}')
```

    Total movements: 193


### Sum up features

We have a really low number of samples to train a model. That's why we should build some features to avoid using a Neural Network model for this time series data.

We're going to use the mean of each attribute for each movement.


```python
numeric_columns = ['xAcceleration', 'yAcceleration', 'zAcceleration', 'pitch', 'yaw', 'roll', 'xRotation', 'yRotation', 'zRotation']
columns = numeric_columns + ['kind']

grouped_movement_data = pd.DataFrame([], columns=columns)

for index, movement in enumerate(splitted_data):
    d = []
    for column in numeric_columns:
        d.append(movement[column].mean())
    d.append(movement['kind'].iloc[0])
    grouped_movement_data = pd.concat([grouped_movement_data, pd.DataFrame([d], columns=columns)], ignore_index=True)
```


```python
grouped_movement_data.groupby('kind')['kind'].hist()
```




    kind
    Backhand    Axes(0.125,0.11;0.775x0.77)
    Drive       Axes(0.125,0.11;0.775x0.77)
    Name: kind, dtype: object




    
![png](README_files/README_24_1.png)
    


### Split data into train and test sets

We're using the method **train_test_split** from **sklearn.model_selection** to automatically select the train and test sets from the whole data. It shuffles the samples and select a given percentage for testing.


```python
X_train, X_test, y_train, y_test = train_test_split(grouped_movement_data[numeric_columns], grouped_movement_data['kind'], test_size=0.2)
y_train.hist()
y_test.hist()
```




    <Axes: >




    
![png](README_files/README_27_1.png)
    


## Build the model

We're going to check the accuracy given from different algorithms: kNN, SVC and Decision Tree.


```python
# kNN model
nbrs = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
y_pred = nbrs.predict(X_test)

print(f'kNN gives an accuracy of {accuracy_score(y_test, y_pred)*100}%')
```

    kNN gives an accuracy of 100.0%



```python
# SVC model
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(f'SVC gives an accuracy of {accuracy_score(y_test, y_pred)*100}%')
```

    SVC gives an accuracy of 100.0%



```python
# Decision Tree model
tree = DecisionTreeClassifier()
tree = tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
y_probab = tree.predict_proba(X_test)

accuracy_score(y_test, y_pred)
print(f'Decision Tree gives an accuracy of {accuracy_score(y_test, y_pred)*100}%')

cm = confusion_matrix(y_test, y_pred, labels=tree.classes_)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
fig, ax = plt.subplots(figsize=(5,5))
cmd.plot(ax=ax)
plt.grid(False)
plt.show()
```

    Decision Tree gives an accuracy of 97.43589743589743%



    
![png](README_files/README_32_1.png)
    


## Conclusion

All three models result in an accuracy near the 100%. Features selected to build these models are classifying correctly each tennis stroke.
