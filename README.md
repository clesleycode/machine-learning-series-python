# Machine Learning with Python


## 1.0 Background

Recall in data structures learning about the different types of tree structures - binary, red black, and splay trees. In tree based modeling, we work off these structures for classification prediction. 

Tree based machine learning is great because it's incredibly accurate and stable, as well as easy to interpret. Despite being a linear model, tree based models map non-linear relationships well. The general structure is as follows: 


## 2.0 Decision Trees

Decision trees are a type of supervised learning algorithm used in classification that works for both categorical and continuous input/output variables. This typle of model includes structures with nodes which represent tests on attributes and the end nodes (leaves) of each branch represent class labels. Between these nodes are what we call edges, which represent a 'decision' that separates the data from the previous node based on some criteria. 

![alt text](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/dt.png "Logo Title Text 1")

Looks familiar, right? 

### 2.1 Nodes

As mentioned above, nodes are an important part of the structure of Decision Trees. In this section, we'll review different types of nodes.

#### 2.1.1 Root Node

The root node is the node at the very top. It represents an entire population or sample because it has yet to be divided by any edges. 

#### 2.1.2 Decision Node

Decision Nodes are the nodes that occur between the root node and leaves of your decision tree. It's considered a decision node because it's a resulting node of an edge that then splits once again into either more decision nodes, or the leaves.

#### 2.1.3 Leaves/Terminal Nodes

As mentioned before, leaves are the final nodes at the bottom of the decision tree that represent a class label in classification. They're also called <i>terminal nodes</i> because more nodes do not split off of them. 

#### 2.1.4 Parent and Child Nodes

A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

### 2.2 Pros & Cons

#### 2.2.1 Pros

1. Easy to Understand: Decision tree output is fairly easy to understand since it doesn't require any statistical knowledge to read and interpret them. Its graphical representation is very intuitive and users can easily relate their hypothesis.

2. Useful in Data exploration: Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. You can refer article (Trick to enhance power of regression model) for one such trick.  It can also be used in data exploration stage. For example, we are working on a problem where we have information available in hundreds of variables, there decision tree will help to identify most significant variable.

3. Less data cleaning required: It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.

4. Data type is not a constraint: It can handle both numerical and categorical variables.

5. Non Parametric Method: Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.

#### 2.2.2 Cons

1. Over fitting: Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning (discussed in detailed below).

2. Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.



In this output, the rows show result for trees with different numbers of nodes. The column `xerror` represents the cross-validation error and the `CP` represents the complexity parameter. 

### 2.2 Pruning Decision Trees

Decision Tree pruning is a technique that reduces the size of decision trees by removing sections (nodes) of the tree that provide little power to classify instances. This is great because it reduces the complexity of the final classifier, which results in increased predictive accuracy by reducing overfitting. 

Ultimately, our aim is to reduce the cross-validation error. First, we index with the smallest complexity parameter:


## 3.0 Random Forests

Recall the ensemble learning method from the Optimization lecture. Random Forests are an ensemble learning method for classification and regression. It works by combining individual decision trees through bagging. This allows us to overcome overfitting. 

### 3.1 Algorithm

First, we create many decision trees through bagging. Once completed, we inject randomness into the decision trees by allowing the trees to grow to their maximum sizes, leaving them unpruned. 

We make sure that each split is based on randomly selected subset of attributes, which reduces the correlation between different trees. 

Now we get into the random forest by voting on categories by majority. We begin by splitting the training data into K bootstrap samples by drawing samples from training data with replacement. 

Next, we estimate individual trees t<sub>i</sub> to the samples and have every regression tree predict a value for the unseen data. Lastly, we estimate those predictions with the formula:

![alt text](https://github.com/lesley2958/ml-tree-modeling/blob/master/rf-pred.png?raw=true "Logo Title Text 1")

where y&#770; is the response vector and x = [x<sub>1</sub>,...,x<sub>N</sub>]<sup>T</sup> &isin; X as the input parameters. 


### 3.2 Advantages

Random Forests allow us to learn non-linearity with a simple algorithm and good performance. It's also a fast training algorithm and resistant to overfitting.

What's also phenomenal about Random Forests is that increasing the number of trees decreases the variance without increasing the bias, so the worry of the variance-bias tradeoff isn't as present. 

The averaging portion of the algorithm also allows the real structure of the data to reveal. Lastly, the noisy signals of individual trees cancel out. 

### 3.3 Limitations 

Unfortunately, random forests have high memory consumption because of the many tree constructions. There's also little performance gain from larger training datasets. 


==========================================================================================


# Machine Learning Optimization

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-pip)
	+ [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
- [2.0 Ensemble Learning](#20-ensemble-learning)
- [3.0 Bagging](#30-bagging)
	+ [3.1 Algorithm](#31-algorithm)
- [4.0 Boosting](#40-boosting)
	+ [4.1 Algorithm](#41-algorithm)
	+ [4.2 Boosting in R](#42-boosting-in-r)
- [5.0 AdaBoosting](#50-adaboosting)
	+ [5.1 Benefits](#51-benefits)
	+ [5.2 Limits](#52-limits)
	+ [5.3 AdaBoost in R](#53-adaboost-in-r)

## 0.0 Setup

TThis guide was written in Python 3.6.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

Let's install the modules we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules:

```
pip3 install scipy
pip3 install numpy
```

## 1.0 Background


## 2.0 Ensemble Learning

Ensemble Learning allows us to combine predictions from different multiple learning algorithms. This is what we consider to be the "ensemble". By doing this, we can have a result with a better predictive performance compared to a single learner.

It's important to note that one drawback is that there's increased computation time and reduces interpretability. 


## 3.0 Bagging

Bagging is a technique where reuse the same training algorithm several times on different subsets of the training data. 

### 3.1 Algorithm

Given a training dataset D of size N, bagging will generate new training sets D<sub>i</sub> of size M by sampling with replacement from D. Some observations might be repeated in each D<sub>i</sub>. 

If we set M to N, then on average 63.2% of the original dataset D is represented, the rest will be duplicates.

The final step is that we train the classifer C on each C<sub>i</sub> separately. 


## 4.0 Boosting

Boosting is an optimization technique that allows us to combine multiple classifiers to improve classification accuracy. In boosting, none of the classifiers just need to be at least slightly better than chance. 

Boosting involves training classifiers on a subset of the training data that is most informative given the current classifiers. 

### 4.1 Algorithm

The general boosting algorithm first involves fitting a simple model to subsample of the data. Next, we identify misclassified observations (ones that are hard to predict). we focus subsequent learners on these samples to get them right. Lastly, we combine these weak learners to form a more complex but accurate predictor.

## 5.0 AdaBoosting

Now, instead of resampling, we can reweight misclassified training examples:

### 5.1 Benefits

Aside from its easy implementation, AdaBoosting is great because it's a simple combination of multiple classifiers. These classifiers can also be different. 

### 5.2 Limits

On the other hand, AdaBoost is sensitive to misclassified points in the training data. 
