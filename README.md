Intro to Machine Learning
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958), [Byte Academy](byteacademy.co), and [ADI](adicu.com).

## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 R and R Studio](#01-r-and-r-studio)
+ [0.2 Virtual Environment](#02-virtual-environment)
- [1.0 Introduction](#10-introduction)
+ [1.1 What is Machine Learning?](#11-what-is-machine-learning)
+ [1.2 Hypothesis](#12-hypothesis)
+ [1.3 Distribution](#13-distribution)
+ [1.4 Notation](#14-notation)
- [2.0 Data](#20-data)
+ [2.1 Labeled vs Unlabeled Data](#21-labeled-vs-unlabeled-data)
+ [2.2 Training vs Test Data](#22-training-vs-test-data)
+ [2.3 Overfitting vs Underfitting](#overfitting-vs-underfitting)
+ [2.4 Open Data](#24-open-data)
- [3.0 Types of Learning](#30-types-of-learning)
+ [3.1 Supervised Learning](#31-supervised-learning)
+ [3.2 Unsupervised Learning](#32-unsupervised-learning)
+ [3.3 Reinforcement Learning](#33-reinforcement-learning)
+ [3.4 Subfields](#34-subfields)
- [4.0 Fundamentals](#40-fundamentals)
+ [4.1 Inputs vs Features](#41-inputs-vs-features)
+ [4.2 Outputs vs Targets](#42-outputs-vs-targets)
+ [4.3 Function Estimation](#43-function-estimation)
* [4.3.1 Exploratory Analysis](#441-exploratory-analysis)
* [4.3.2 Data Visualization](#432-data-visualization)
* [4.3.3 Linear Model](#433-linear-model)
+ [4.4 Bias and Variance](#44-bias-and-variance)
* [4.4.1 Bias](#441-bias)
* [4.4.2 Variance](#442-variance)
* [4.4.3 Bias-Variance Tradeoff](#443-bias-variance-tradeoff)
- [5.0 Optimization](#50-optimization)
+ [5.1 Loss Function](#51-loss-function)
+ [5.2 Boosting](#52-boosting)
* [5.2.1 What is a weak learner?](#521-what-is-a-weak-learner)
* [5.2.2 What is an ensemble?](#522-what-is-an-ensemble)
+ [5.3 Occam's Razor](#53-occams-razor)
- [6.0 Final Words](#60-final-words)
+ [6.1 Resources](#61-resources)
+ [6.2 Mini Courses](#62-mini-courses)

## 0.0 Setup

This guide was written in R 3.2.3.

### 0.1 R and R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("ggplot2")
install.packages("caret")
```

### 0.2 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows:

```
pip3 install virtualenv
virtualenv your_env
```

And then launch it with:

```
source your_env/bin/activate
```

## 1.0 Introduction

### 1.1 What is Machine Learning?

Machine Learning is the field where statistics and computer science overlap for prediction insights on data. What do we mean by <i>prediction</i>? Given a dataset, we generate an algorithm to <i>learn</i> what features or attributes are indicators of a certain label or prediction. These attributes or features can be anything that describes a data point, whether that's height, frequency, class, etc. 

This algorithm is chosen based on the original dataset, which you can think of as prior or historical data. When we refer to machine learning algorithms, we're referring to a function that best maps a data point's features to its label. A large part of the machine learning is spent improving this function as much as possible. 

### 1.2 Hypothesis

You've likely heard of hypotheses in the context of science before; typically, it's an educated guess on what and outcome will be. In the context of machine learning, a hypothesis is the function we believe is similar to the <b>true</b> function of learning - the target function that we want to model and use as our machine learning algorithm. 

### 1.3 Assumptions 

In this course, we'll review different machine learning algorithms, from decision trees to support vector machines. Each is different in its methodology and results. A critical part of the process of choosing the best algorithm is considering the assumptions you can make about the particular dataset you're working with. These assumptions can include the linearity or lack of linearity, the distribution of the dataset, and much more. 

### 1.4 Notation

In this course and future courses, you'll see lots of notation. I've collected a list of all the general notation you'll see:

<i>a</i>: scalar <br>
<i><b>a</b></i>: vector <br>
<i>A</i>: matrix <br>
<i>a<sub>i</sub></i>: the ith entry of <i><b>a</b></i> <br>
<i>a<sub>ij</sub></i>: the entry (i,j) of <i>A</i> <br>
<i><b>a<sup>(n)</sup></b></i>: the nth vector <i><b>a</b></i> in a dataset <br>
<i>A<sup>(n)</sup></i>: the nth matrix <i>A</i> in a dataset <br>
<i>H</i>: Hypothesis Space, the set of all possible hypotheses <br>


## 2.0 Data 

As a data scientist, knowing the different forms data takes is highly important. When working on a prediction problem, the collection of data you're working with is called a <i>dataset</i>.

### 2.1 Labeled vs Unlabeled Data

Generally speaking, there are two forms of data: unlabeled and labeled. Labeled data refers to data that has inputs and attached, known outputs. Unlabeled means all you have is inputs. They're denoted as follows:

- Labeled Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>, Y = {y<sup>n</sup> &isin; R}<sup>N</sup><sub>n=1</sub>

- Unlabed Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>

Here, X denotes a feature set containing N samples. Each of these samples is a d-dimension vector, <b>x<sup>(n)</sup></b>. Each of these dimensions is an attribute, feature, variable, or element. Meanwhile, Y is the label set. 

### 2.2 Training vs Test Data

When it comes time to train your classifier or model, you're going to need to split your data into <b>testing</b> and <b>training</b> data. 

Typically, the majority of your data will go towards your training data, while only 10-25% of your data will go towards testing. 

It's important to note there is no overlap between the two. Should you have overlap or use all your training data for testing, your accuracy results will be wrong. Any classifier that's tested on the data it's training is obviously going to do very well since it will have observed those results before, so the accuracy will be high, but wrongly so. 

Furthermore, both of these sets of data must originate from the same source. If they don't, we can't expect that a model built for one will work for the other. We handle this by requiring the training and testing data to be <b>identically and independently distributed (iid)</b>. This means that the testing data show the same distribution as the training data, but again, must not overlap.

Together these two aspects of the data are known as <i>IID assumption</i>.

### 2.3 Overfitting vs Underfitting

The concept of overfitting refers to creating a model that doesn't generaliz e to your model. In other words, if your model overfits your data, that means it's learned your data <i>too</i> much - it's essentially memorized it. 

This might not seem like it would be a problem at first, but a model that's just "memorized" your data is one that's going to perform poorly on new, unobserved data. 

Underfitting, on the other hand, is when your model is <i>too</i> generalized to your data. This model will also perform poorly on new unobserved data. This usually means we should increase the number of considered features, which will expand the hypothesis space. 

### 2.4 Open Data 

What's open data, you ask? Simple, it's data that's freely  for anyone to use! Some examples include things you might have already heard of, like APIs, online zip files, or by scraping data!

You might be wondering where this data comes from - well, it can come from a variety of sources, but some common ones include large tech companies like Facebook, Google, Instagram. Others include large institutions, like the US government! Otherwise, you can find tons of data from all sorts of organizations and individuals. 


## 3.0 Types of Learning

Generally speaking, Machine Learning can be split into three types of learning: supervised, unsupervised, and reinforcement learning. 

### 3.1 Supervised Learning

This algorithm consists of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, Decision Tree, Random Forest, Logistic Regression, etc.

All supervised learning algorithms in the Python module scikit-learn hace a `fit(X, y)` method to fit the model and a `predict(X)` method that, given unlabeled observations X, returns predicts the corresponding labels y.

### 3.2 Unsupervised Learning

In this algorithm, we do not have any target or outcome variable to predict / estimate.  We can derive structure from data where we don't necessarily know the effect of the variables. Examples of Unsupervised Learning: Clustering, Apriori algorithm, K-means.

### 3.3 Reinforcement Learning

Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: Markov Decision Process.

### 3.4 Subfields

Though Machine Learning is considered the overarching field of prediction analysis, it's important to know the distinction between its different subfields.

#### 3.4.1 Natural Language Processing

Natural Language Processing, or NLP, is an area of machine learning that focuses on developing techniques to produce machine-driven analyses of textual data.

#### 3.4.2 Computer Vision

Computer Vision is an area of machine learning and artificial intelligence that focuses on the analysis of data involving images.

#### 3.4.3 Deep Learning

Deep Learning is a branch of machine learning that involves pattern recognition on unlabeled or unstructured data. It uses a model of computing inspired by the structure of the brain, which we call this model a neural network.


## 4.0 Fundamentals

### 4.1 Inputs vs Features

The variables we use as inputs to our machine learning algorithms are commonly called inputs, but they are also frequently called predictors or features. Inputs/predictors are independent variables and in a simple 2D space, you can think of an input as x-axis variable.

### 4.2 Outputs vs Targets

The variable that we’re trying to predict is commonly called a target variable, but they are also called output variables or response variables. You can think of the target as the dependent variable; visually, in a 2-dimensional graph, the target variable is the variable on the y-axis.

#### 4.2.1 Label Types

Another name for outputs or targets is **labels**. In this section we discuss the different types of labels a dataset can encompass. 

1. Single column with binary values -- usually a classification problem where a sample belongs to one class only and there are only two classes. 
2. Single column with real values -- usually a regression problem with a prediction of only one value.
3. Multiple columns with binary values -- also usually a classification problem, where a sample belongs to one class, but there are more than two classes. 
4. Multiple columns with real values -- also usually a regression problem with a prediction of multiple values.
5. Multilabel - also a classification problem, but a sample can belong to several classes.

### 4.3 Function Estimation

When you’re doing machine learning, specifically supervised learning, you’re using computational techniques to reverse engineer the underlying function your data. 

With that said, we'll go through what this process generally looks like. In the exercise, I intentionally keep the underlying function hidden from you because we never know the underlying function in practice. Once again, machine learning provides us with a set of statistical tools for estimate <i>f(x)</i>.

So we begin with getting our data:

``` R
df.unknown_fxn_data <- read.csv(url("http://www.sharpsightlabs.com/wp-content/uploads/2016/04/unknown_fxn_data.txt"))
```

#### 4.3.1 Exploratory Analysis

Now we’ll perform some basic data exploration. First, we’ll use `str()` to examine the “structure” of the data:

``` R
str(df.unknown_fxn_data)
```

Which gets us two variables, `input_var` and `target_var`. 

``` bash
'data.frame':	28 obs. of  2 variables:
$ input_var : num  -2.75 -2.55 -2.35 -2.15 -1.95 -1.75 -1.55 -1.35 -1.15 -0.95 ...
$ target_var: num  -0.224 -0.832 -0.543 -0.725 -1.026 ...
```

Next, let’s print out the first few rows of the dataset using the head() function.

``` R
head(df.unknown_fxn_data)
```

You can think of the two columns as (x,y) pairs.

#### 4.3.2 Data Visualization

Now, let’s examine the data visually. Since `input_var` is an independent variable, we can plot it on the x-axis and `target_var` on the y-axis. And since these are (x, y) pairs, we'll use a scatterplot to visualize it.

``` R
require(ggplot2)
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point()
```

This is important because by plotting the data visually, we can visually detect a pattern which we can later use to compare to our estimated function.

#### 4.3.3 Linear Model

Now that we’ve explored our data, we’ll create a very simple linear model using caret. Here, the `train()` function is the “core” function of the caret package, which builds the machine learning model. 

Inside of the `train()` function, we’re using `df.unknown_fxn_data` for the data parameter. `target_var ~ input_var` specifies the “formula” of our model and indicates the target variable that we’re trying to predict as well as the predictor we’ll use as an input. The `method = "lm"` indicates that we want a linear regression model. 

``` R
require(caret)

mod.lm <- train( target_var ~ input_var
,data = df.unknown_fxn_data
,method = "lm"
)
```

Now that we’ve built the simple linear model using the `train()` function, let’s plot the data and plot the linear model on top of it.

To do this, we’ll extract the slope and the intercept of our linear model using the `coef()` function.

``` R
coef.icept <- coef(mod.lm$finalModel)[1]
coef.slope <- coef(mod.lm$finalModel)[2]
```

The following plots the data:

``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point() +
geom_abline(intercept = coef.icept, slope = coef.slope, color = "red")
```

As I mentioned in the beginning of this example, I kept the underlying function hidden, which was a sine function with some noise. When we began this exercise, there was a hidden function that generated the data and we used machine learning to estimate that function.

``` R
set.seed(9877)
input_var <- seq(-2.75,2.75, by = .2)
target_var <- sin(input_var) + rnorm(length(input_var), mean = 0, sd = .2)
df.unknown_fxn_data <- data.frame(input_var, target_var)
```

Here, we just visualize that underlying function. 

``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
geom_point() +
stat_function(fun = sin, color = "navy", size = 1)
```

### 4.4 Bias and Variance

Understanding how different sources of error lead to bias and variance helps us improve the data fitting process resulting in more accurate models.

#### 4.4.1 Bias

Error due to bias is the difference between the expected (or average) prediction of our model and the actual value we're trying to predict. 

#### 4.4.2 Variance

Error due to variance is taken as the variability of a model prediction for a given data point. In other words, the variance is how much the predictions for a given point vary between different realizations of the model.

![alt text](https://github.com/lesley2958/intro-ml/blob/master/biasvar.png?raw=true "Logo Title Text 1")

#### 4.4.3 Bias-Variance Tradeoff

The bias–variance tradeoff is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.

## 5.0 Optimization

In the simplest case, optimization consists of maximizing or minimizing a function by systematically choosing input values from within an allowed set and computing the value of the function. 

### 5.1 Loss Function

The job of the loss function is to tell us how inaccurate a machine learning system's prediction is in comparison to the truth. It's denoted with &#8467;(y, y&#770;), where y is the true value and y&#770; is a the machine learning system's prediction. 

The loss function specifics depends on the type of machine learning algorithm. In Regression, it's (y - y&#770;)<sup>2</sup>, known as the squared loss. Note that the loss function is something that you must decide on based on the goals of learning. 

Since the loss function gives us a sense of the error in a machine learning system, the goal is to <i>minimize</i> this function. Since we don't know what the distribution does, we have to calculte the loss function for each data point in the training set, so it becomes: 

![alt text](https://github.com/lesley2958/intro-ml/blob/master/trainingerror.png?raw=true "Logo Title Text 1")

In other words, our training error is simply our average error over the training data. Again, as we stated earlier, we can minimize this to 0 by memorizing the data, but we still want it to generalize well so we have to keep this in mind when minimizing the loss function. 

### 5.2 Boosting

Boosting is a machine learning meta-algorithm that iteratively builds an ensemble of weak learners to generate a strong overall model.

#### 5.2.1 What is a weak learner?

A <i>weak learner</i> is any machine learning algorithm that has an accuracy slightly better than random guessing. For example, in binary classification, approximately 50% of the samples belong to each class, so a weak learner would be any algorithm that slightly improves this score – so 51% or more. 

These weak learners are usually fairly simple because using complex models usually leads to overfitting. The total number of weak learners also needs to be controlled because having too few will cause underfitting and have too many can also cause overfitting.

#### 5.2.2 What is an ensemble?

The overall model built by Boosting is a weighted sum of all of the weak learners. The weights and training given to each ensures that the overall model yields a pretty high accuracy.

#### 5.2.3 What do we mean by iteratively build?

Many ensemble methods train their components in parallel because the training of each those weak learners is independent of the training of the others, but this isn't the case with Boosting. 

At each step, Boosting tries to evaluate the shortcomings of the overall model built, and then generates a weak learner to battle these shortcomings. This weak learner is then added to the total model. Therefore, the training must necessarily proceed in a serial/iterative manner.

Each of the iterations is basically trying to improve the current model by introducing another learner into the ensemble. Having this kind of ensemble reduces the bias and the variance. 

#### 5.2.4 Gradient Boosting


#### 5.2.5 AdaBoost


#### 5.2.6 Disadvantages

Because boosting involves performing so many iterations and generating a new model at each, a lot of computations, time, and space are utilized.

Boosting is also incredibly sensitive to noisy data. Because boosting tries to improve the output for data points that haven't been predicted well. If the dataset has misclassified or outlier points, then the boosting algorithm will try to fit the weak learners to these noisy samples, leading to overfitting. 

### 5.3 Occam's Razor

Occam's Razor states that any phenomenon should make as few assumptions as possible. Said again, given a set of possible solutions, the one with the fewest assumptions should be selected. The problem here is that Machine Learning often puts accuracy and simplicity in conflict.


## 6.0 Final Words

What we've covered here is a very small glimpse of machine learning. Applying these concepts in actual machine learning algorithms such as random forests, regression, classification, is the harder but doable part. 

### 6.1 Resources

[Stanford Coursera](https://www.coursera.org/learn/machine-learning) <br>
[Math &cap; Programming](https://jeremykun.com/main-content/)

### 6.2 Mini Courses

Learn about courses [here](http://www.byteacademy.co/all-courses/data-science-mini-courses/).

[Python 101: Data Science Prep](https://www.eventbrite.com/e/python-101-data-science-prep-tickets-30980459388) <br>
[Intro to Data Science & Stats with R](https://www.eventbrite.com/e/data-sci-109-intro-to-data-science-statistics-using-r-tickets-30908877284) <br>
[Data Acquisition Using Python & R](https://www.eventbrite.com/e/data-sci-203-data-acquisition-using-python-r-tickets-30980705123) <br>
[Data Visualization with Python](https://www.eventbrite.com/e/data-sci-201-data-visualization-with-python-tickets-30980827489) <br>
[Fundamentals of Machine Learning and Regression Analysis](https://www.eventbrite.com/e/data-sci-209-fundamentals-of-machine-learning-and-regression-analysis-tickets-30980917759) <br>
[Natural Language Processing with Data Science](https://www.eventbrite.com/e/data-sci-210-natural-language-processing-with-data-science-tickets-30981006023) <br>
[Machine Learning with Data Science](https://www.eventbrite.com/e/data-sci-309-machine-learning-with-data-science-tickets-30981154467) <br>
[Databases & Big Data](https://www.eventbrite.com/e/data-sci-303-databases-big-data-tickets-30981182551) <br>
[Deep Learning with Data Science](https://www.eventbrite.com/e/data-sci-403-deep-learning-with-data-science-tickets-30981221668) <br>
[Data Sci 500: Projects](https://www.eventbrite.com/e/data-sci-500-projects-tickets-30981330995)


Machine Learning with Naive Bayes
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-pip)
- [1.0 Introduction](#10-introduction)
	+ [1.1 Bayes Theorem](#11-bayes-theorem)
	+ [1.2 Overview](#12-overview)
		* [1.2.1 Frequency Table](#121-frequency-table)
		* [1.2.2 Likelihood Table](#122-likelihood-table)
		* [1.2.3 Calculation](#123-calculation)
  + [1.3 Naive Bayes Evaluation](#13-naive-bayes-evaluation)
    * [1.3.1 Pros](#131-pros)
    * [1.3.2 Cons](#132-cons)
- [2.0 Naive Bayes Types](#20-naive-bayes-types)
	+ [2.1 Gaussian](#21-gaussian)
	+ [2.2 Multinomial](#22-multinomial)
	+ [2.3 Bernoulli](#23-bernoulli)
  + [2.4 Tips for Improvement](#24-tips-for-improvement)
- [3.0 Sentiment Analysis](#30-sentiment-analysis)
	+ [3.1 Preparing the Data](#31-preparing-the-data)
		* [3.1.1 Training Data](#311-training-data)
		* [3.1.2 Test Data](#312-test-data)
	+ [3.2 Building a Classifier](#32-building-a-classifier)
	+ [3.3 Classification](#33-classification)
	+ [3.4 Accuracy](#34-accuracy)


  
## Setup

This guide was written in Python 3.6.

### Python and Pip

If you haven't already, please download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

Let's install the modules we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules: 

```
pip3 install scikit-learn==0.18.1
pip3 install nltk==3.2.4
```

## Introduction

In this tutorial set, we'll review the Naive Bayes Algorithm used in the field of machine learning. Naive Bayes works on Bayes Theorem of probability to predict the class of a given data point, and is extremely fast compared to other classification algorithms. 

Because it works with an assumption of independence among predictors, the Naive Bayes model is easy to build and particularly useful for large datasets. Along with its simplicity, Naive Bayes is known to outperform even some of the most sophisticated classification methods.

This tutorial assumes you have prior programming experience in Python and probablility. While I will overview some of the priciples in probability, this tutorial is **not** intended to teach you these fundamental concepts. If you need some background on this material, please see my tutorial [here](https://github.com/lesley2958/intro-stats).


### Bayes Theorem

Recall Bayes Theorem, which provides a way of calculating the *posterior probability*: 

![alt text](https://github.com/lesley2958/intro-stats/blob/master/images/bayes.png?raw=true)

Before we go into more specifics of the Naive Bayes Algorithm, we'll go through an example of classification to determine whether a sports team will play or not based on the weather. 

To start, we'll load in the data, which you can find [here](https://github.com/lesley2958/ml-bayes/blob/master/data/weather.csv).


```python
import pandas as pd
f1 = pd.read_csv("./data/weather.csv")
```

Before we go any further, let's take a look at the dataset we're working with. It consists of 2 columns (excluding the indices), *weather* and *play*. The *weather* column consists of one of three possible weather categories: `sunny`, `overcast`, and `rainy`. The *play* column is a binary value of `yes` or `no`, and indicates whether or not the sports team played that day.


```python
f1.head(3)
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weather</th>
      <th>Play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Overcast</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rainy</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



#### Frequency Table

If you recall from probability theory, frequencies are an important part of eventually calculating the probability of a given class. In this section of the tutorial, we'll first convert the dataset into different frequency tables, using the `groupby()` function. First, we retrieve the frequences of each combination of weather and play columns: 


```python
df = f1.groupby(['Weather','Play']).size()
print(df)
```

    Weather   Play
    Overcast  Yes     4
    Rainy     No      3
              Yes     2
    Sunny     No      2
              Yes     3
    dtype: int64


It will also come in handy to split the frequencies by weather and yes/no. Let's start with the three weather frequencies:


```python
df2 = f1.groupby('Weather').count()
print(df2)
```

              Play
    Weather       
    Overcast     4
    Rainy        5
    Sunny        5


And now for the frequencies of yes and no:


```python
df1 = f1.groupby('Play').count()
print(df1)
```

          Weather
    Play         
    No          5
    Yes         9


#### Likelihood Table

The frequencies of each class are important in calculating the likelihood, or the probably that a certain class will occur. Using the frequency tables we just created, we'll find the likelihoods of each weather condition and yes/no. We'll accomplish this by adding a new column that takes the frequency column and divides it by the total data occurances:


```python
df1['Likelihood'] = df1['Weather']/len(f1)
df2['Likelihood'] = df2['Play']/len(f1)
print(df1)
print(df2)
```

          Weather  Likelihood
    Play                     
    No          5    0.357143
    Yes         9    0.642857
              Play  Likelihood
    Weather                   
    Overcast     4    0.285714
    Rainy        5    0.357143
    Sunny        5    0.357143


Now, we're able to use the Naive Bayesian equation to calculate the posterior probability for each class. The highest posterior probability is the outcome of prediction.


#### Calculation

Now, let's get back to our question: *Will the team play if the weather is sunny?*

From this question, we can construct Bayes Theorem. Because the *know* factor is that it is sunny, the $P(A \mid B)$ becomes $P(Yes \mid Sunny)$. From there, it's just a matter of plugging in probabilities. 

![Screen Shot 2017-08-17 at 3.17.44 PM.png](https://steemitimages.com/DQmPxCqp6fZKA4jvuByPuh9jMAejcXg2huD7J2v3Nt4P6J1/Screen%20Shot%202017-08-17%20at%203.17.44%20PM.png)

Since we already created some likelihood tables, we can just index `P(Sunny)` and `P(Yes)` off the tables:


```python
ps = df2['Likelihood']['Sunny']
py = df1['Likelihood']['Yes']
```

That leaves us with $P(Sunny \mid Yes)$. This is the probability that the weather is sunny given that the players played that day. In `df`, we see that the total number of `yes` days under `sunny` is 3. We take this number and divide it by the total number of `yes` days, which we can get from `df`:


```python
psy = df['Sunny']['Yes']/df1['Weather']['Yes']
```

And finally, we can just plug these variables into bayes theorem: 


```python
p = (psy*py)/ps
print(p)
```

    0.6


This tells us that there's a 60% likelihood of the team playing if it's sunny. Because this is a binary classification of yes or no, a value greater than 50% indicates a team *will* play. 



### 1.3 Naive Bayes Evaluation

Every classifier has pros and cons, whether that be in terms of computational power, accuracy, etc. In this section, we'll review the pros and cons of Naive Bayes.

#### 1.3.1 Pros

Naive Bayes is incredibly easy and fast in predicting the class of test data. It also performs well in multi-class prediction.

When the assumption of independence is true, the Naive Bayes classifier performs better thanother models like logistic regression. It does this, and with less need of a lot of data.

Naive Bayes also performs well with categorical input variables compared to numerical variable(s), which is why we're able to use it for text classification. For numerical variables, normal distribution must be assumed.

#### 1.3.2 Cons

If a categorical variable has a category not observed in the training data set, then model will assign a 0 probability and will be unable to make a prediction. This is referred to as “Zero Frequency”. To solve this, we can use the smoothing technique, such as Laplace estimation.

Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.


## 2.0 Naive Bayes Types

With `scikit-learn`, we can implement Naive Bayes models in Python. There are three types of Naive Bayes models, all of which we'll review in the following sections.

### 2.1 Gaussian

The Gaussian Naive Bayes Model is used in classification and assumes that features will follow a normal distribution. 

We begin an example by importing the needed modules:

``` python
from sklearn.naive_bayes import GaussianNB
import numpy as np
```

As always, we need predictor and target variables, so we assign those:

``` python
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])

y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
```

Now we can initialize the Gaussian Classifier:

``` python
model = GaussianNB()
```

Now we can train the model using the training sets:
``` python
model.fit(x, y)
```

Now let's try out an example:
``` python
predicted = model.predict([[1,2],[3,4]])
```

We get:
```
([3,4])
```

### 2.2 Multinomial

MultinomialNB implements the multinomial Naive Bayes algorithm and is one of the two classic Naive Bayes variants used in text classification. This classifier is suitable for classification with discrete features (such as word counts for text classification). 

The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts may also work.

First, we need some data, so we import numpy and 

``` python
import numpy as np
x = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
```

Now let's build the Multinomial Naive Bayes model: 

``` python3
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x, y)
```

Let's try an example:

``` python
print(clf.predict(x[2:3]))
```

And we get:

```
[3]
```

### 2.3 Bernoulli

Like MultinomialNB, this classifier is suitable for discrete data. BernoulliNB implements the Naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions, meaning there may be multiple features but each one is assumed to be a binary value. 

The decision rule for Bernoulli Naive Bayes is based on

![alt text](https://github.com/lesley2958/ml-bayes/blob/master/bernoulli.png?raw=true "Logo Title Text 1")


``` python
import numpy as np
x = np.random.randint(2, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])
```

``` python
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(x, y)
print(clf.predict(x[2:3]))
```

And we get: 
```
[3]
```

### 2.4 Tips for Improvement

If continuous features don't have a normal distribution (recall the assumption of normal distribution), you can use different methods to convert it to a normal distribution.

As we mentioned before, if the test data set has a zero frequency issue, you can apply smoothing techniques “Laplace Correction” to predict the class.

As usual, you can remove correlated features since the correlated features would be voted twice in the model and it can lead to over inflating importance.


## 3.0 Sentiment Analysis  

So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

### 3.1 Preparing the Data 

To accomplish sentiment analysis computationally, we have to use techniques that will allow us to learn from data that's already been labeled. 

So what's the first step? Formatting the data so that we can actually apply NLP techniques. 

``` python
import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
```

Here, format_sentence changes a piece of text, in this case a tweet, into a dictionary of words mapped to True booleans. Though not obvious from this function alone, this will eventually allow us to train  our prediction model by splitting the text into its tokens, i.e. <i>tokenizing</i> the text.

``` 
{'!': True, 'animals': True, 'are': True, 'the': True, 'ever': True, 'Dogs': True, 'best': True}
```

You'll learn about why this format is important is section 3.2.

Using the data on the github repo, we'll actually format the positively and negatively labeled data.

``` python
pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
```

``` python
neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
```


#### 3.1.1 Training Data

Next, we'll split the labeled data we have into two pieces, one that can "train" data and the other to give us insight on how well our model is performing. The training data will inform our model on which features are most important.

``` python
training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
```

#### 3.1.2 Test Data

We won't use the test data until the very end of this section, but nevertheless, we save the last 10% of the data to check the accuracy of our model. 
``` python
test = pos[int((.1)*len(pos)):] + neg[int((.1)*len(neg)):]
```

### 3.2 Building a Classifier

``` python
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
```

All NLTK classifiers work with feature structures, which can be simple dictionaries mapping a feature name to a feature value. In this example, we’ve used a simple bag of words model where every word is a feature name with a value of True. This is an implementation of the Bernoulli Naive Bayes Model.
 
To see which features informed our model the most, we can run this line of code:

```python
classifier.show_most_informative_features()
```

```
Most Informative Features
        no = True                neg : pos    =     20.6 : 1.0
    awesome = True               pos : neg    =     18.7 : 1.0
    headache = True              neg : pos    =     18.0 : 1.0
   beautiful = True              pos : neg    =     14.2 : 1.0
        love = True              pos : neg    =     14.2 : 1.0
          Hi = True              pos : neg    =     12.7 : 1.0
        glad = True              pos : neg    =      9.7 : 1.0
       Thank = True              pos : neg    =      9.7 : 1.0
         fan = True              pos : neg    =      9.7 : 1.0
        lost = True              neg : pos    =      9.3 : 1.0
```

### 3.3 Classification

Just to see that our model works, let's try the classifier out with a positive example: 

```python
example1 = "this workshop is awesome."

print(classifier.classify(format_sentence(example1)))
```

```
'pos'
```

Now for a negative example:

``` python
example2 = "this workshop is awful."

print(classifier.classify(format_sentence(example2)))
```

```
'neg'
```

### 3.4 Accuracy

Now, there's no point in building a model if it doesn't work well. Luckily, once again, nltk comes to the rescue with a built in feature that allows us find the accuracy of our model.

``` python
from nltk.classify.util import accuracy
print(accuracy(classifier, test))
```

``` 
0.9562326869806094
```

Turns out it works decently well!

## 4.0 Joint Models

If you have input data x and want to classify the data into labels y. A <b>generative model</b> learns the joint probability distribution `p(x,y)` and a discriminative model learns the <b>conditional probability distribution</b> `p(y|x)`.

Here's an simple example of this form:

```
(1,0), (1,0), (2,0), (2, 1)
```

`p(x,y)` is

```
      y=0   y=1
     -----------
x=1 | 1/2   0
x=2 | 1/4   1/4
```

Meanwhile,

```
p(y|x) is
```

```
      y=0   y=1
     -----------
x=1 | 1     0
x=2 | 1/2   1/2
```

Notice that if you add all 4 probabilities in the first chart, they add up to 1, but if you do the same for the second chart, they add up to 2. This is because the probabilities in chart 2 are read row by row. Hence, `1+0=1` in the first row and `1/2+1/2=1` in the second.

The distribution `p(y|x)` is the natural distribution for classifying a given example `x` into a class `y`, which is why algorithms that model this directly are called discriminative algorithms. 

Generative algorithms model `p(x, y)`, which can be tranformed into `p(y|x)` by applying Bayes rule and then used for classification. However, the distribution `p(x, y)` can also be used for other purposes. For example you could use `p(x,y)` to generate likely `(x, y)` pairs.


## 5.0 Final Words

### 5.1 Resources

Machine Learning with Classification
==================

Brought to you by [Lesley Cordero](lesley2958.github.io).

## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 R and R Studio](#01-r-and-r-studio)
+ [0.2 Python and Pip](#02-python-and-pip)
+ [0.3 Virtual Environment](#03-virtual-environment)
- [1.0 Introduction](#10-introduction)
- [2.0 Nearest Neighbor](#20-nearest-neighbor)
+ [2.1 k-Nearest Neighbor Classifier](#21-k-nearest-neighbor-classifier)
- [3.0 Support Vector Machines](#30-support-vector-machines)
+ [3.1 Separating Hyperplane](#31-separating-hyperlanes)
* [3.1.1 Optimal Separating Hyperplane](#311-optimal-separating-hyperplane)
+ [3.2 Margins](#32-margins)
+ [3.3 Equation](#33-equations)
* [3.3.1 Example](#331-example)
* [3.3.2 Margin Computation](#332-margin-computation)
- [4.0 Iris Classification](#40-iris-classification)
+ [4.1 Kernels](#41-kernels)
+ [4.2 Tuning](#42-tuning)
* [4.2.1 Gamma](#421-gamma)
* [4.2.2 C value](#422-c-value)
+ [4.3 Evaluation](#43-evaluation)
* [4.3.1 Pros](#431-pros)
* [4.3.2 Cons](#432-cons)


## 0.0 Setup

This guide was written in R 3.2.3.

### 0.1 R and R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("RTextTools")
```

### 0.2 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

Then, on your command line, install the needed modules as follows:

``` 
pip3 install sklearn
```

### 0.3 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows: 
```
pip3 install virtualenv
virtualenv your_env
```
And then launch it with: 
```
source your_env/bin/activate
```

## 1.0 Introduction

Classification is a machine learning algorithm whose output is within a set of discrete labels. This is in contrast to Regression, which involved a target variable that was on a continuous range of values.

Throughout this tutorial we'll review different classification algorithms, but you'll notice that the workflow is consistent: split the dataset into training and testing datasets, fit the model on the training set, and classify each data point in the testing set. 


## 2.0 LDA vs QDA

Discriminant Analysis is a statistical technique used to classify data into groups based on each data point's features. 

``` R
library(dplyr)
library(magrittr)
olives <- readRDS("olives.RDS")

split <- createDataPartition(y=olives$Region,
p=0.80,
list=FALSE)
train <- olives[split,]
test <- olives[-split,]
```

In R, there's a library called `MASS` that provides us with built-in functions to perform discriminant analysis. 

``` R
library(MASS)
```

### 2.1 Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is a linear classification algorithm used for when it can be assumed that the covariance is the same for <b>all</b> classes. Its mostly used as a dimension reduction technique in the pre-processing portion so that the different classes are separable and therefore easier to classify and avoid overfitting. 

LDA is similar to Principal Component Analysis (PCA), but LDA also considers the axes that maximize the separation between multiple classes. It works by finding the linear combinations of the original variables that gives the best separation between the groups in the data set. 

``` R
lda <- lda(train$Region ~ ., data=train[,3:10])
plot(lda)
```

``` R
train_predict <- predict(lda, train[,3:10])
(t <- table(train$Region, train_predict$class))
test_predict <- predict(lda, test[,3:10])
(t <- table(test$Region, test_predict$class))
```

### 2.2 Quadratic Discriminant Analysis

Quadratic Discriminant Analysis, on the other hand, is used for heterogeneous variance-covariance matrices. Because QDA has more parameters to estimate, it's typically less accurate than LDA. 

``` R
qda <- qda(train$Region ~ ., data=train[,3:10])
```

``` R
train_predict <- predict(qda, train[,3:10])
(t <- table(train$Region, train_predict$class))
test_predict <- predict(qda, test[,3:10])
(t <- table(test$Region, test_predict$class))
```

## 3.0 Support Vector Machines

Support Vector Machines (SVMs) is a machine learning algorithm used for classification tasks. Its primary goal is to find an optimal separating hyperplane that separates the data into its classes. This optimal separating hyperplane is the result of the maximization of the margin of the training data.

### 3.1 Separating Hyperplane

Let's take a look at the scatterplot of the iris dataset that could easily be used by a SVM: 

![alt text](https://github.com/lesley2958/regression/blob/master/log-scatter.png?raw=true "Logo Title Text 1")

Just by looking at it, it's fairly obvious how the two classes can be easily separated. The line which separates the two classes is called the <b>separating hyperplane</b>.

In this example, the hyperplane is just two-dimensional, but SVMs can work in any number of dimensions, which is why we refer to it as <i>hyperplane</i>.

#### 3.1.1 Optimal Separating Hyperplane

Going off the scatter plot above, there are a number of separating hyperplanes. The job of the SVM is find the <i>optimal</i> one. 

To accomplish this, we choose the separating hyperplane that maximizes the distance from the datapoints in each category. This is so we have a hyperplane that generalizes well.

### 3.2 Margins

Given a hyperplane, we can compute the distance between the hyperplane and the closest data point. With this value, we can double the value to get the <b>margin</b>. Inside the margin, there are no datapoints. 

The larger the margin, the greater the distance between the hyperplane and datapoint, which means we need to <b>maximize</b> the margin. 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/margin.png?raw=true "Logo Title Text 1")

### 3.3 Equation

Recall the equation of a hyperplane: w<sup>T</sup>x = 0. Here, `w` and `x` are vectors. If we combine this equation with `y = ax + b`, we get:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/wt.png?raw=true "Logo Title Text 1")

This is because we can rewrite `y - ax - b = 0`. This then becomes:

w<sup>T</sup>x = -b &Chi; (1) + (-a) &Chi; x + 1 &Chi; y

This is just another way of writing: w<sup>T</sup>x = y - ax - b. We use this equation instead of the traditional `y = ax + b` because it's easier to use when we have more than 2 dimensions.  

#### 3.3.1 Example

Let's take a look at an example scatter plot with the hyperlane graphed: 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/ex1.png?raw=true "Logo Title Text 1")

Here, the hyperplane is x<sub>2</sub> = -2x<sub>1</sub>. Let's turn this into the vector equivalent:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/vectex1.png?raw=true "Logo Title Text 1")

Let's calculate the distance between point A and the hyperplane. We begin this process by projecting point A onto the hyperplane.

![alt text](https://github.com/lesley2958/ml-classification/blob/master/projex1.png?raw=true "Logo Title Text 1")

Point A is a vector from the origin to A. So if we project it onto the normal vector w: 

![alt text](https://github.com/lesley2958/ml-classification/blob/master/normex1.png?raw=true "Logo Title Text 1")

This will get us the projected vector! With the points (3,4) and (2,1) [this came from w = (2,1)], we can compute `||p||`. Now, it'll take a few steps before we get there.

We begin by computing `||w||`: 

`||w||` = &#8730;(2<sup>2</sup> + 1<sup>2</sup>) = &#8730;5. If we divide the coordinates by the magnitude of `||w||`, we can get the direction of w. This makes the vector u = (2/&#8730;5, 1/&#8730;5).

Now, p is the orthogonal prhoojection of a onto w, so:

![alt text](https://github.com/lesley2958/ml-classification/blob/master/orthproj.png?raw=true "Logo Title Text 1")

#### 3.3.2 Margin Computation

Now that we have `||p||`, the distance between A and the hyperplane, the margin is defined by:

margin = 2||p|| = 4&#8730;5. This is the margin of the hyperplane!


## 4.0 Iris Classification

Let's perform the iris classification from earlier with a support vector machine model:

``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target
```

Now we can create an instance of SVM and fit the data. For this to happen, we need to declare a regularization parameter, C. 

``` python
C = 1.0 
svc = svm.SVC(kernel='linear', C=1, gamma=1).fit(X, y)
```

Based off of this classifier, we can create a mesh graph:

``` python
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
```

Now we pull the prediction method on our data:

``` python
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
```

Lastly, we visualize this with matplotlib:

``` python
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
```

### 4.1 Kernels

Classes aren't always easily separable. In these cases, we build a decision function that's polynomial instead of linear. This is done using the <b>kernel trick</b>.

![alt text](https://github.com/lesley2958/regression/blob/master/svm.png?raw=true "Logo Title Text 1")

If you modify the previous code with the kernel parameter, you'll get different results!

``` python
svc = svm.SVC(kernel='rbf', C=1, gamma=1).fit(X, y)
```

Usually, linear kernels are the best choice if you have large number of features (>1000) because it is more likely that the data is linearly separable in high dimensional space. Also, you can RBF but do not forget to cross validate for its parameters as to avoid overfitting.

### 4.2 Tuning

Tuning parameters value for machine learning algorithms effectively improves the model performance. Let’s look at the list of parameters available with SVM.

#### 4.2.1 Gamma 

Notice our gamma value from earlier at 1. The higher the value of gamma the more our model will try to exact fit the as per training data set, leading to an overfit model.

Let's see our visualizations with gamma values of 10 and 100: 

``` python
svc = svm.SVC(kernel='linear', C=1, gamma=10).fit(X, y)
svc = svm.SVC(kernel='linear', C=1, gamma=100).fit(X, y)
```

#### 4.2.2 C value

Penalty parameter C of the error term. It also controls the trade off between smooth decision boundary and classifying the training points correctly. Let's see the effect when C is increased to 10 and 100:

``` python
svc = svm.SVC(kernel='linear', C=10, gamma=1).fit(X, y)
svc = svm.SVC(kernel='linear', C=100, gamma=1).fit(X, y)
```

Note that we should always look at the cross validation score to have effective combination of these parameters and avoid over-fitting.

### 4.3 Evaluation

As with any other machine learning model, there are pros and cons. In this section, we'll review the pros and cons of this particular model.

#### 4.3.1 Pros

SVMs largest stength is its effectiveness in high dimensional spaces, even when the number of dimensions is greater than the number of samples. Lastly, it uses a subset of training points in the decision function (called support vectors), so it's also memory efficient.

#### 4.3.2 Cons

Now, if we have a large dataset, the required time becomes high. SVMs also perform poorly when there's noise in the dataset. Lastly, SVM doesn’t directly provide probability estimates and must be calculated using an expensive five-fold cross-validation.


