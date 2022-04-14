# Machine Learning with Python Classification: TUTORIAL

**For reference**: 
Main: https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec
Others:
- Pandas: https://pandas.pydata.org/docs/reference/index.html
- Matplotlib: https://matplotlib.org/3.5.0/index.html
- Seaborn: https://seaborn.pydata.org/index.html

**Author**: Mauro Di Pietro

**Prerequisite**: 
- Approaching ML with no fundamental knowledge with **Data Analysis**, start from beginning.
- Know basic Python
- Quite long post with more explanation as opposed to the original tutorial. But if patient enough, we will get all skills and understandings we need
- I will demo on **Colab** of course. But some explanatino would be in pure IDE

# Let's explore things:

Since this tutorial can be a good starting point for beginners, we will use the **“Titanic dataset”** from the famous **Kaggle competition**, in which you are provided with passengers data and the task is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” 

Link download: https://www.kaggle.com/competitions/titanic/overview

Download 3 files (I also include in this repo): 

![image](https://user-images.githubusercontent.com/48288606/163347645-f6c1b260-9014-414a-857c-5730cea4654a.png)

We will present some useful Python code that can be easily used in other similar cases (just copy, paste, run) and walk through every line of code with comments, so that we can easily replicate this example 

Link full code [Titanic Data Classification]()

|Table of Content|
|----------------|
|Environment setup: import libraries and read data|
|Data Analysis: understand the meaning and the predictive power of the variables|
|Feature engineering: extract features from raw data|
|Preprocessing: data partitioning, handle missing values, encode categorical variables, scale|
|Feature Selection: keep only the most relevant variables|
|Model design: train, tune hyperparameters, validation, test|
|Performance evaluation: read the metrics|
|Explainability: understand how the model produces results|

## Setup

```python
## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular
```

Try reading the data into a **pandas Dataframe**.

```
dtf = pd.read_csv('train.csv')
dtf.head()
```

![image](https://user-images.githubusercontent.com/48288606/163349291-c4ddada1-5832-40c4-9c9d-bb43aa975dc3.png)

- **read_csv()** will read .csv files and return a DataFrame object (full content of a csv file)
- **head()** returns the first n rows. (Default is 5)

Data and type of the return value: 

<p align="center"><img width=450 height=300 src="https://user-images.githubusercontent.com/48288606/163351010-ba964fca-ff67-4d7b-bea8-61372ac02043.png"></p>

Each row of the table represents a specific passenger (or observation). If you are working with a different dataset that doesn’t have a structure like that, in which each row represents an observation, then you need to summarize data and transform it.

Now that it’s all set, we will start by analyzing data, then select the features, build a machine learning model and predict.

## Data Analysis

In statistics, [exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) is the process of summarizing the main characteristics of a dataset to understand what the data can tell us beyond the formal modeling or hypothesis testing task.

I always start by getting an overview of the whole dataset, in particular I want to know how many categorical and numerical variables there are and the proportion of missing data. Recognizing a variable’s type sometimes can be tricky because categories can be expressed as numbers (the Survived column is made of 1s and 0s). To this end, I am going to write a simple function that will do that for us:

```python
'''
Recognize whether a column is numerical or categorical.
:parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical or "num" otherwise
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"
```

- **dtype** return the type of a column
- **nunique()** return number of distinct elements in specified column.

Try with "Sex" column:

<p align="center"><img width=400 height=70 src="https://user-images.githubusercontent.com/48288606/163354565-88cb8e74-f439-43bd-beb3-c60c2ceafd27.png"></p>

We define that if **type** is **object** or **number of unique values** is **less than 20**. It's **category**, else it's **number**. In most cases, **category** column does not contain more than 20 distinctive values. That's why we define like this.

This function is very useful and can be used in several occasions. To give an illustration I’ll plot a [heatmap](https://en.wikipedia.org/wiki/Heat_map) of the dataframe to visualize columns type and missing data.

```python
dic_cols = {col:utils_recognize_type(dtf, col, max_cat=20) for col in dtf.columns}
heatmap = dtf.isnull()
for k,v in dic_cols.items():
 if v == "num":
   heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
 else:
   heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=True).set_title('Dataset Overview')
plt.show()
print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")
```

- isnull() return a new DataFrame indicating specific data cells is null (True) or not (False) 
- apply() allows manipulate each data cells in one [Series](https://pandas.pydata.org/docs/reference/series.html)(for specified column_name) with our function 
- sns.heatmap(heat_map_name) plots a heatmap
- plt.show() displays all **open** figures (make sure that the event loop is running to have responsive figures - we've already open with **sns.heatmap()**)
- print() with [ANSI escape code format](https://en.wikipedia.org/wiki/ANSI_escape_code) (_"escape_code\[style_code;background_color_code; fore_ground_color_code String_to_be_printed"_ - **default escape_code is "\033"**)

This code will create a dictionary with _{key:value}_ that is _{column_name:column_type}_ . Then with each row of each column, if this column is "number", "heat index" should be 0.5 _(0 <= index <= 1)_ for missing-value cell, else it's 1. Similar to "category" column, slightly change the "heat level" for "null" cells is 0, just for differentiating between 2 types of columns.

Experimental result:

<p align="center"><img width=400 height=70 src="https://user-images.githubusercontent.com/48288606/163360731-1e41aea3-f723-41dc-8ef8-96f508830b8a.png"></p>

Analysing:

- Each row of the table represents a specific passenger (or observation) identified by PassengerId, so I’ll set it as index (or primary key of the table for SQL lovers).
- Survived is the phenomenon that we want to understand and predict (or target variable), so I’ll rename the column as “Y”. It contains two classes: 1 if the passenger survived and 0 otherwise, therefore this use case is a binary classification problem.
- Age and Fare are numerical variables while the others are categorical.
- Only Age and Cabin contain missing data.

Set columns: 

```python
dtf = dtf.set_index("PassengerId")
dtf = dtf.rename(columns={"Survived":"Y"})
```

We believe visualization is the best tool for data analysis, but you need to know what kind of plots are more suitable for the different types of variables. Therefore, I’ll provide the code to plot the appropriate visualization for different examples.

First, let’s have a look at the univariate distributions (probability distribution of just one variable). A bar plot is appropriate to understand labels frequency for a single categorical variable. For example, let’s plot the target variable:

```python
y = "Y"
ax = dtf[y].value_counts().sort_values().plot(kind="barh")
totals= []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
     ax.text(i.get_width()+.3, i.get_y()+.20, 
     str(round((i.get_width()/total)*100, 2))+'%', 
     fontsize=10, color='black')
ax.grid(axis="x")
plt.suptitle(y, fontsize=20)
plt.show()
```
