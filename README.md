# Machine Learning (+ Data Analysis) with Python Classification: TUTORIAL

**For reference**: 

Main: https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec <br>
Others:
- Pandas: https://pandas.pydata.org/docs/reference/index.html
- Matplotlib: https://matplotlib.org/3.5.0/index.html
- Seaborn: https://seaborn.pydata.org/index.html

**Author**: Mauro Di Pietro

**Prerequisite**: 
- Approaching ML with no fundamental knowledge with **Data Analysis**, start from beginning.
- Knowing basic **Python**
- Quite long post with more explanation about library like `seaborn`, `mathplotlib`, `pandas`, `numpy`, ... and its function for those are "newbie" in this aspect, as opposed to the original tutorial. But if patient enough, we will get all skills and understandings we need
- I will demo on **Colab** of course. But some explanations would be in pure IDE

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

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/163360731-1e41aea3-f723-41dc-8ef8-96f508830b8a.png"></p>

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

- **value_counts()** returns a Series containing counts of unique values.
- **sort_values()** sort by the values with ascending order (default)
- **patches** property are [Artists](https://matplotlib.org/stable/api/artist_api.html#matplotlib.artist.Artist) with a face color and an edge color. [More info](https://matplotlib.org/stable/api/patches_api.html). Consider this as an object to render for the chart
- **text()** method add text to the Axes. View used parameters [here](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.text.html)
- **grid()** configures the ruler for the chart. [More info](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.grid.html)
- **suptitle()** adds title for the chart
- **show()** displays the chart.

Some notable propertes and methods (test with "Sex" column):

```python
column_name = "Sex"
value_count = dtf[column_name].value_counts()

print(value_count)
print("Type of return value: " + str(type(value_count)))

print("========= Sorted values ============")
ordered_value_count = dtf[column_name].value_counts().sort_values()
print(ordered_value_count)
print("Type of return value: " + str(type(ordered_value_count)))

plot = dtf[column_name].value_counts().sort_values().plot(kind="barh") # Horizontal bar chart
print("Type of plot: " + str(type(str(plot))))
for item in plot.patches:
    print("Object with format(Type, coordinate, width, height, angle): " + str(item))
```
Output: 

<p align="center"><img width=900 height=225 src="https://user-images.githubusercontent.com/48288606/163439859-732b07c2-4711-40bc-8e6e-9a9879977edc.png"></p>

Experimental result:

<p align="center"><img  src="https://user-images.githubusercontent.com/48288606/163438265-5064fe3d-ad56-4b59-8255-30e66781257f.png"></p>

The survival rate (or the population mean) is 38%.

Moreover, a [histogram](https://en.wikipedia.org/wiki/Histogram) is perfect to give a rough sense of the density of the underlying distribution of a single numerical data. I recommend using a [box plot](https://en.wikipedia.org/wiki/Box_plot) to graphically depict data groups through their quartiles. Let’s take the Age variable for instance:

```python
x = "Age"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
fig.suptitle(x, fontsize=20)
### distribution
ax[0].title.set_text('distribution')
variable = dtf[x].fillna(dtf[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(True)
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot 
ax[1].title.set_text('outliers (log scale)')
tmp_dtf = pd.DataFrame(dtf[x])
tmp_dtf[x] = np.log(tmp_dtf[x])
tmp_dtf.boxplot(column=x, ax=ax[1])
plt.show()
```

- **subplots()** creates many sub-chart in the current figure. [More info](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
- **suptitle()** sets the title below the chart
- **fillna()** would fill the null's cell with defined method. In this case, we use mean() method which returns a mean(average) number of specific column to fill the "holes"
- **linspace()** returns evenly spaced numbers over a specified interval. [More info](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
- **quantile()** return one or array of quantiles. [Main document](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html). This consumes me lots of time to understand, take time ! More references:
  - [Quantile Method NumPy](https://www.skytowner.com/explore/numpy_quantile_method)
  - [Quantile Definition](https://www.youtube.com/watch?v=IFKQLDmRK0Y&ab_channel=StatQuestwithJoshStarmer)
  -  > NOTE: The default interpolation is **linear**.
- **distplot()** combines the seaborn [kdeplot()](https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot) and [rugplot()](https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot) functions, plots univariate or bivariate distributions . [More info](https://seaborn.pydata.org/generated/seaborn.distplot.html)
- **axvline()** adds a vertical line across the Axes. [More info](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.axvline.html)
- **description()** generates descriptive statistics.
- **log()** returns natural logarithm 

Some notable propertes and methods (test with "Age" column):
 
```python
column_name = "Age"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False) # Return tuple (figure, [array of Axes])
print("Figure: " + str(fig) )
print("Axes: " + str(ax) + " - " +  str(len(ax)) + " Sub-plot" )

variable  = dtf[column_name].fillna(dtf[column_name].mean()) # DataFrame 's handled with null values
breaks = np.quantile(variable, q=np.linspace(0, 1, 11)) # Create array of quatile points with divisor is 10.

print("Array of quantile point: " + str(breaks))  

des = dtf[column_name].describe()
print("=========== Statistics's information =========\n" + str((des)))
```

Output:

<p align="center"><img width=900 height=300 src="https://user-images.githubusercontent.com/48288606/163519338-81a10528-5be1-47ba-9e86-cb533227d8a7.png"></p>

Experimental result:

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/163519391-3f4a568d-ef78-4fe0-bd7d-0576b8158381.png"></p>

The passengers were, on average, pretty young: the distribution is skewed towards the left side (the mean is 30 y.o and the 75th percentile is 38 y.o.). Coupled with the outliers in the box plot, the first spike in the left tail says that there was a significant amount of children.

## Feature Engineering

It’s time to create new features from raw data using domain knowledge. We will provide one example: We'll try to create a useful feature by extracting information from the `Cabin` column. We're assuming that the letter at the beginning of each cabin number (i.e. **"B96"**) indicates some kind of section, maybe there were some lucky sections near to lifeboats. I will summarize the observations in clusters by extracting the section of each cabin:

```python
## Create new column
dtf["Cabin_section"] = dtf["Cabin"].apply(lambda x: str(x)[0])

## Plot contingency table
cont_table = pd.crosstab(index=dtf["Cabin_section"], 
             columns=dtf["Pclass"], values=dtf["Y"], aggfunc="sum")
sns.heatmap(cont_table, annot=True, cmap="YlGnBu", fmt='.0f',
            linewidths=.5).set_title( 
            'Cabin_section vs Pclass (filter: Y)' )
```

- **crosstab()** Compute a simple cross tabulation of two factors `Cabin_section` and `Pclass`. Computes with a function defined by `aggfunc` which is `sum`. Hence, sum the total of columns with given value specified in `values` which is `dtf["Y"]` ([crosstab reference](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)). Hard to understand? For example:

```python
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

dtf = pd.read_csv("train.csv")

dtf = dtf.set_index("PassengerId") # Set index for "PassengerId"
dtf = dtf.rename(columns={"Survived":"Y"}) # Rename column "Survived" to "Y"

dtf["Cabin_section"] = dtf["Cabin"].apply(lambda x: str(x)[0])
cont_table = pd.crosstab(index=dtf["Cabin_section"], 
             columns=dtf["Pclass"], values=dtf["Y"], aggfunc="sum")
print(cont_table)

```

Output about the passenger with  the relationship between `Cabin_section` and `Pclass` and their death:

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165130865-842927eb-8e13-4ce8-ad99-a1327fe7d225.png"></p>

Then plotting **heatmap** with:

```python
sb.heatmap(cont_table, annot=True, cmap="YlGnBu", fmt='.0f').set_title( 'Cabin_section vs Pclass (filter: Y)' )
plt.show()
```

- `cont_table` is our dataframe
- `cmap` indicates color (optional), [more color](https://python-graph-gallery.com/92-control-color-in-seaborn-heatmaps). Some most-used colors (we use the first):

```python
# plot using a color palette
sns.heatmap(df, cmap="YlGnBu")
plt.show()

sns.heatmap(df, cmap="Blues")
plt.show()

sns.heatmap(df, cmap="BuPu")
plt.show()

sns.heatmap(df, cmap="Greens")
plt.show()
```
<p align="center"><img width=800 height=550 src="https://user-images.githubusercontent.com/48288606/165232289-694ab4c5-e80a-4c82-a109-4f7d313fd23e.png"></p>

- `fmt` (format string when adding annotations, explained [here](https://stackoverflow.com/questions/54506626/how-to-understand-seaborns-heatmap-annotation-format)).<br> [Heatmap reference](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

Experimental result (just as the same as the figure above with adding colors :> ):

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165132114-d2af9153-5842-4432-ab66-95e29ca1bcc8.png"></p>

This plot shows how survivors are distributed among cabin sections and classes (7 survivors are in section A, 35 in B, …). Most of the sections are assigned to the 1st and the 2nd classes, while the majority of missing sections ("n") belongs to the 3rd class. We are going to keep this new feature instead of the column Cabin:

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165230410-7db9b687-ffa8-4217-a3f3-97e14b839918.png"></p>

## Preprocessing

Data preprocessing is the phase of preparing the raw data to make it suitable for a machine learning model. In particular:
1. Each observation must be represented by a single row, in other words you can’t have two rows describing the same passenger because they will be processed separately by the model (the dataset is already in such form, so ✅). Moreover, each column should be a feature, so you shouldn’t use PassengerId as a predictor, that’s why this kind of table is called “feature matrix”.
2. The dataset must be partitioned into at least two sets: the model shall be trained on a significant portion of your dataset (so-called “train set”) and tested on a smaller set (“test set”).
3. Missing values should be replaced with something, otherwise your model may freak out.
4. Categorical data must be encoded, which means converting labels into integers, because machine learning expects numbers not strings.
5. It’s good practice to scale the data, it helps to normalize the data within a particular range and speed up the calculations in an algorithm.

Alright, let’s begin by partitioning the dataset. When splitting data into train and test sets you must follow 1 basic rule: rows in the train set shouldn’t appear in the test set as well. That’s because the model sees the target values during training and uses it to understand the phenomenon. In other words, the model already knows the right answer for the training observations and testing it on those would be like cheating. I’ve seen a lot of people pitching their machine learning models claiming 99.99% of accuracy that did in fact ignore this rule. Luckily, the Scikit-learn package knows that:

```python
## split data
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
## print info
print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())
```

`model_selection.train_test_split` randomizes and divides our dataframe (current `train.csv` dataset) to 2 part, _test set_ covers **0.3** (defined with `test_size`) and _train set_ covers **the rest (0.7)**. We have **891 rows**; therefore, there will be **623 rows** for _train set_ and **268 ones** for _test set_ <br>
More reference: [train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

```python
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)

print("======== Train Dataset ===========")
print(dtf_train)

print("======== Test Dataset ===========")
print(dtf_test)
```

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165315838-a1ab5534-beef-4331-bd8b-141f0abe8d37.png"></p>

`drop()` will remove a specific columns with specified name `Y` from dataframe and `shape()` will return a tuple with the numbers of rows and columns (rows, columns). `mean()` with second parameter is the number of order of `axis` (for example, with 2-dimension array, `0` is along columns, `1` is along rows) 

Experimental result:

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165321827-a10e658a-c772-46c6-af0c-e0fed42660b6.png"></p>

Next step: the `Age` column contains some **missing data (19%)** that need to be handled. In practice, you can replace missing data with a specific value, like _9999_, that keeps trace of the missing information but _changes the variable distribution_. Alternatively, you can use _the average_ of the column, like I’m going to do. I’d like to underline that from a **Machine Learning** perspective, it’s correct to first split into train and test and then replace **N/A** values with the average of the training set only.

There are still some categorical data that should be **encoded**. The two most common encoders are the [Label-Encoder](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/) (each unique label is mapped to an integer) and the [One-Hot-Encoder](https://en.wikipedia.org/wiki/One-hot) (each label is mapped to a binary vector). The first one is suited for data with **ordinality** only. If applied to a column with no ordinality, like `Sex`, it would turn the vector [male, female, female, male, …] into [1, 2, 2, 1, …] and we would have that **female > male** and with an average of 1.5 which makes no sense. On the other hand, the _One-Hot-Encoder_ would transform the previous example into two dummy [variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) (\***dichotomous** quantitative variables): Male [1, 0, 0, 1, …] and Female [0, 1, 1, 0, …]. It has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space, but _features with high cardinality_ can lead to a [dimensionality issue](https://en.wikipedia.org/wiki/Curse_of_dimensionality#:~:text=The%20curse%20of%20dimensionality%20refers,was%20coined%20by%20Richard%20E.). We shall use the _One-Hot-Encoding_ method, transforming 1 categorical column with **n** unique values into **n-1** dummies. Conveniently, in **pandas**, we have [pandas.get_dummies()](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) function which help us to handle that. Let’s encode `Sex` as an example:

> Note: `dichotomous` mentions completely opposing ideas or things

```python
## create dummy
dummy = pd.get_dummies(dtf_train["Sex"], prefix="Sex",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="Sex", axis=1).head() )

## drop the original categorical column
dtf = dtf_train.drop("Sex", axis=1)
```

`get_dummies()` converts categorical variable into dummy/indicator variables and creates columns with _header name_ of indicated string "Sex" in `prefix` combining with each value . `drop_first` is a option to choose whether to get k-1 dummies out of k categorical levels by removing the first level, in this case is `Sex_female`. This just for simplifying our ouput, because we totally calculate `Sex_female` (the "first" columns) based on the rest, we may not need this.

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165437651-f92e501e-d450-4433-8c86-bc56642e3810.png"></p>

Experimental result:

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165328729-aaccccb3-0821-4048-9003-5a87d525e455.png"></p>

We do similary, use _One-Hot-Encoding_ with `Embarked` and `Pclass`:

**Embark**:

```python
## create dummy
dummy = pd.get_dummies(dtf_train["Embarked"], prefix="Embarked",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="Embarked", axis=1).head() )

## drop the original categorical column
dtf = dtf_train.drop("Embarked", axis=1)
```

Experimental result (remove `Embarked_C`):

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165435106-3c9f6228-c44b-4b03-ab84-7f513d8f06d6.png"></p>

**Pclass:** 

```python
## create dummy
dummy = pd.get_dummies(dtf_train["Pclass"], prefix="Pclass",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="Pclass", axis=1).head() )

## drop the original categorical column
dtf = dtf_train.drop("Pclass", axis=1)
```

Experimental result (remove `Pclass_1`):

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165438134-97479520-a9da-440a-9028-15d4c9b36b27.png"></p>


That's *One-hot encoding* with `Sex`. Last but not least, we're going to scale the features. There are several different ways to do that, I’ll present just the most used ones: the **Standard-Scaler** and the **MinMax-Scaler**. The first one assumes data is normally distributed and rescales it such that the distribution centres around 0 with a standard deviation of 1. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values, therefore this scaler can’t guarantee _balanced feature scales in the presence of outliers_. On the other hand, the _MinMax-Scaler_ rescales the data set such that all feature values are in the same range (0–1). It is less affected by outliers but compresses all inliers in a narrow range. Since my data is not normally distributed, We'll go with the _MinMax-Scaler_. But before, we gonna remove **Categorical** columns (which contain characters or strings), keep columns with "dummies" or "number".

```python
dtf_train = dtf_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Cabin_section'], axis=1)
dtf_train
```

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165439089-0a76f9d6-8ac5-4988-a6ac-f7a7fda5fa50.png"></p>

Let's go and rescale our feature with _MinMax-Scaler_ :

<p align="center"><img src="https://user-images.githubusercontent.com/48288606/165439467-e1b316e5-dcb7-4c1a-b1f6-7af74d5be0bd.png"></p>

## Feature Selection

Feature selection is the process of selecting a subset of relevant variables to build the machine learning model. It makes the model easier to interpret and reduces overfitting (when the model adapts too much to the training data and performs badly outside the train set).
I already did a first “manual” feature selection during data analysis by excluding irrelevant columns. Now it’s going to be a bit different because we assume that all the features in the matrix are relevant and we want to drop the unnecessary ones. When a feature is not necessary? Well, the answer is easy: when there is a better equivalent, or one that does the same job but better.
I’ll explain with an example: Pclass is highly correlated with Cabin_section because, as we’ve seen before, certain sections were located in 1st class and others in the 2nd. Let’s compute the correlation matrix to see it:
