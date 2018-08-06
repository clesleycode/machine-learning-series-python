# Machine Learning with Python


========================================================================

# Unsupervised Machine Learning

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-pip)
	+ [0.1 Other](#01-other)
- [1.0 Introduction](#10-introduction)
- [6.0 Final Words](#50-final-words)

## 0.0 Setup

This guide was written in Python 3.6.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

Let's install the modules we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules: 

```
pip3 install time
pip3 install sklearn
```

## 1.0 Introduction

As we've covered before, there are two general categories that machine learning falls into. First is supervised learning, which we've covered with regression analysis, decision trees, and support vector machines. 

Recall that supervised learning is when your explanatory variables X come with an target variable Y. In contrast, unsupervised learning has no labels, so we a lot of X's with no Y's. In unsupervised learning all we can do is try our best to extract some meaning out of the data's underlying structure and do some checks to make sure that our methods are robust.

### 1.1 Clustering 

One example of an unsupervised learning algorithm is clustering! Clustering is exactly what it sounds like. It's a way of grouping “similar” data points together into clusters or subgroups, while keeping each group as distinct as possible. 

In this way data points belonging to different clusters will be quite different from each other, too. This is useful because oftentimes we'll come across datasets which exhibit this kind of grouped structure. Now, you might be thinking how are two points considered similar? That's a fair point and there are two ways in which we determine that: 1. Similarity 2. Cluster centroid. We'll go into detail on what these two things mean in the next section. 

### 1.2 Similarity 

Intuitively, it makes sense that similar things should be close to each other, while different things should be farther apart. So to formalize the notion of similarity, we choose a distance metric (see below) that can quantify exactly how "close" two points are to each other. The most commonly used distance metric is the Euclidean distance which we should all be pretty familiar with (think: distance formula from middle school), and that's what we'll be using in our example today. We'll introduce some other distance metrics towards the end.

### 1.3 Cluster Centroid

The cluster centroid is the most representative feature of the entire cluster. We say "feature" instead of "point" because the centroid may not necessarily be an existing point in the cluster. You can find it by averaging the values of all the points belonging to a specific group. But any relevant information about the cluster centroid tells us everything that we need to know about all other points in the same cluster.


## 2.0 K Means Clustering

The k-means algorithm has a simple objective: given a set of data points, it tries to separate them out into k distinct clusters. It uses the same principle that we mentioned earlier: keep the data points within each cluster as similar as possible. You have to provide the value of k to the algorithm, so you should have a general idea of how many clusters you're expecting to see in your data. This sin't a precise science, but we can utilize visualization techniques to help us choose a proper k. 

So let’s begin by doing just that. Remember that clustering is an unsupervised learning method, so we’re never going to have a perfect answer for our final clusters, but we'll do our best to make sure that the results we get are reasonable and replicable. 

By replicable, we mean that our results can be arrived at by someone else using a different starting point. By reasonable, we mean that our results have to show some correlation with what we expect to encounter in real life.

The following image is just an example of the visualization we might get. Notice the three colors and the ways in which they could be separated, so we can set k to 3. Right now we’re operating under the assumption that we know how many clusters we want, but we’ll go into more detail about relaxing this assumption and how to choose the best possible k at the end of the workshop.

![alt text](https://camo.githubusercontent.com/6e540cb12555953bf43925fc20d46b6da1768017/687474703a2f2f707562732e7273632e6f72672f73657276696365732f696d616765732f525343707562732e65506c6174666f726d2e536572766963652e46726565436f6e74656e742e496d616765536572766963652e7376632f496d616765536572766963652f41727469636c65696d6167652f323031322f414e2f6332616e3136313232622f6332616e3136313232622d66332e676966 "Logo Title Text 1")

### 2.1 Centroid Initialization

First we initialize three random cluster centroids. We initialize these clusters randomly because every iteration of k-means will "correct" them towards the right clusters. Since we are heading to a correct answer anyway, we don't really care about where we start.

As we explained before, these centroids are our “representative points” -- they contain all the information that we need about other points in the same cluster. It makes sense to think about these centroids as being the physical center of each cluster, so let’s pretend like our randomly initialized cluster centers are the actual centroids, and group our points accordingly. Here we use our distance metric of choice, in this case the Euclidean distance. So for every single data point we have, we compute the two distances: one from the first cluster centroid, and the other from the second centroid. We assign this data point to the cluster at which the distance to the centroid is the smallest. This makes sense, because intuitively we’re grouping points which are closer together.


### 2.2 Cluster Formation

Now we have something that’s starting to resemble three distinct clusters! But remember that we need to update the centroids that we started with -- we’ve just added in a bunch of new data points to each cluster, so we need our “representative point,” or our centroid, to reflect that.

So we’ll just do quick averaging of all the values within each cluster and call that our new centroid. The new centroids are further "within" the data than the older centroids. Notice that we’re not quite done yet -- we have some straggling points which don’t really seem to belong in either cluster. Let’s run another iteration of k-means and see if that separates out the clusters better. So recall that we’re just computing the distances from the centroids for each data point, and re-assigning those that are closer to centroids of the other cluster.


### 2.3 Iteration

We keep computing the centroids for every iteration using the steps before. After doing the few iterations, maybe you’ll notice that the clusters don’t change after a certain point. This actually turns out to be a good criterion for stopping the cluster iterations! At that point we’re just wasting time and computational resources. So let’s formalize this idea of a “stopping criterion.” We define a small value, &epsilon;, and we can terminate the algorithm when the change in cluster centroids is less than epsilon. This way, epsilon serves as a measure of how much error we can tolerate.


## 3.0 Image Segmentation

Now we'll move onto a k-means example with images! 

Images often have a few dominant colors -- for example, the bulk of the image is often made up of the foreground color and the background color. In this example, we'll write some code that uses scikit-learn's k-means clustering implementation to find the what these dominant colors may be.

Once we know what the most important colors are in an image, we can compress (or "quantize") the image by re-expressing the image using only the set of k colors that we get from the algorithm. We'll be analyzing the two following images:

![alt text](https://github.com/adicu/AccessibleML/blob/master/datasets/kmeans/imgs/leo_bb.png?raw=true "Logo Title Text 1")

![alt text](https://github.com/adicu/AccessibleML/blob/master/datasets/kmeans/imgs/mario.png?raw=true "Logo Title Text 1")

We'll be using the following modules, so make sure to import them:

``` python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
```

Then we begin this exercise by reading in the image as a matrix and normalizing it:

``` python
img = mpimg.imread("./leo.png")
img = img * 1.0 / img.max()

```

An image is represented here as a three-dimensional array of floating-point numbers, which can take values from 0 to 1. If we look at `img.shape`, we'll find that the first two dimensions are x and y, and then the last dimension is the color channel. There are three color channels (one each for red, green, and blue). A set of three channel values at a single (x, y)-coordinate is referred to as a "pixel".

``` python
width, height, num_channels = img.shape
num_pixels = width * height
```

We're going to use a small random sample of 10% of the image to find our clusters:

``` python
num_sample_pixels = num_pixels / 10
```

Next we need to reshape the image data into a single long array of pixels (instead of a two-dimensional array of pixels) in order to take our sample.

``` python
img_reshaped = np.reshape(img, (num_pixels, num_channels))
img_sample = shuffle(img_reshaped, random_state=0)
```

Now that we have our data, let's construct our k-means object and feed it some data. It will find the best k clusters, as determined by a distance function. We're going to try to find the 20 colors which best represent the colors in the picture, so we set k to 20:

``` python
K = 20
```

Here, we're instantiating the kmeans object just as we have done with other machine learning models. the t0 is initialized to track how fast this algorithm takes to fit, which is the next step in this process. Lastly, we just print how long it took. Note: this code has to be run at the same time so we can get an accurate estimate of how long it took!

``` python
t0 = time()
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(img_sample)
print("K-means clustering complete. Elapsed time: {} seconds".format(time() - t0))
```

The centers of each of the clusters represents a color that was significant in the image. We can grab the values of these colors from kmeans.cluster_centers_. We can also call kmeans.predict() to match each pixel in the image to the closest color, which will let us know the size of each cluster (and also serve as a way to quantize the image)

``` python
kmeans.cluster_centers_
```

As you can see, there are K cluster centers, each of which is a RGB color
```
array([[ 0.59594023,  0.37377197,  0.23699242],
[ 0.07824585,  0.06161205,  0.04534107],
[ 0.98117697,  0.98098147,  0.97990966],
[ 0.29123059,  0.28127983,  0.22996978],
[ 0.88017613,  0.67859817,  0.51104909],
[ 0.52016801,  0.51832098,  0.43664253],
[ 0.37142357,  0.3536061 ,  0.2840144 ],
[ 0.27072299,  0.13391563,  0.05868939],
[ 0.97850031,  0.85233569,  0.70350802],
[ 0.47486466,  0.27632579,  0.16393569],
[ 0.67109919,  0.48759544,  0.33595228],
[ 0.01431993,  0.01277512,  0.01052323],
[ 0.47390169,  0.43343523,  0.33971789],
[ 0.21901846,  0.21610278,  0.17843075],
[ 0.57573938,  0.59766436,  0.536443  ],
[ 0.17545493,  0.07990164,  0.03137593],
[ 0.3704485 ,  0.19818981,  0.1059725 ],
[ 0.15592512,  0.15792704,  0.13496453],
[ 0.93447602,  0.77906561,  0.61215806],
[ 0.78221744,  0.57008743,  0.40647745]], dtype=float32)
```

Now, we can predict on sample pixels and see how long that takes:
``` python
t0 = time()
labels = kmeans.predict(img_reshaped)
print("k-means labeling complete. Elapsed time: {} seconds".format(time() - t0))
```
You should get an answer under a second! Next, we can construct a histogram of the points in each cluster:

``` python
n, bins, patches = plt.hist(labels, bins=range(K+1))
for p, color in zip(patches, kmeans.cluster_centers_):
plt.setp(p, 'facecolor', color)
```

![alt text](https://github.com/lesley2958/ml-unsupervised/blob/master/hist.png?raw=true "Logo Title Text 1")


As you might be able to tell from the above histogram, the most dominant color in the scene is the background color, followed by a large drop down to the foreground colors. This isn't all that surprising, since visually we can see that the space is mostly filled with the background color -- that's why it's called the "background".

Now, let's redraw the scene using only the cluster centers. This can be used for image compression, since we only need to store the index into the list of cluster centers and the colors corresponding to each center, rather than the colors corresponding to each pixel in the image.

``` python
quantized_img = np.zeros(img.shape)
for i in range(width):
for j in range(height):
# We need to do some math here to get the correct
# index position in the labels array
index = i * height + j
quantized_img[i][j] = kmeans.cluster_centers_[labels[index]]

quantized_imgplot = plt.imshow(quantized_img)
```

![alt text]( "Logo Title Text 1")

Notice that the image looks similar, but that the gradients are no longer as smooth and there are a few image artifacts scattered throughout. This is because we're only using the k best colors, which excludes the steps along the gradient.

## 4.0 Limitations and Extensions

In our very first example, we started with k = 3 centroids. In case you're wondering how we arrived at this magic number and why, read on.

### 4.1 Known Number of Centroids 

Sometimes, you may be in a situation where the number of clusters is provided to you beforehand. For example, you may be asked to categorize a vast range of different bodily actions to the three main subdivisions of the brain (cerebrum, cerebellum and medulla). 

Here you know that you are looking for three main clusters where each cluster will represent the part of the brain the data point is grouped to. So in this situation, you expect to have three centroids.


### 4.2 Unknown Number of Centroids

However, there may be other situations while training in which you may not even know how many centroids to pick up from your data. Two extreme situations generally happen.

*
#### 4.2.1 Extreme Cases

You could either end up making each point its own representative (a perfect centroid) at the risk of losing any grouping tendencies. This is usually called the overfitting problem. While each point perfectly represents itself, it gives you no general information about the data as a whole and will be unable to tell you anything relevant about new data that is coming in.

You could end up choosing only one centroid from all the data (a perfect grouping). Since there is no way to generalize an enormous volume of data to one point alone, this method loses relevant distinguishing features of the data.This is kind of like saying that all the people in the world drink water, so we can cluster them all by this feature. In Machine Learning terminology, this is called the underfitting problem. Underfitting implies that we are generalizing all of our data to a potentially trivial common feature.

#### 4.2.2 Stability

Unfortunately, there’s no easy way to determine the optimal value of k. It’s a hard problem: we have to think about balancing out the number of clusters that makes the most sense for our data, while at the same time making sure that we don’t overfit our model to the exact dataset that we have. There are a few ways that we can address this, and we’ll briefly mention them here.

The most intuitive explanation is the idea of stability. If the clusters we obtain represent a true, underlying pattern in our data, it makes sense that the clusters shouldn’t change very much on separate but similar samples. So if we randomly subsample or split our data into smaller parts and run the clustering algorithm again, the cluster memberships shouldn’t drastically change. If they did, that’d be an indication that our clusters were too finely-tuned to the random noise in our data. Therefore, we can compute “stability scores” for a fixed value of k and observe which value of k gives us the most stable clusters. This idea of perturbation is really important for machine learning in general, and will come up time and time again.

We can also use penalization approaches, where we use different criterion such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) to keep the value of k under control.


==========================================================================================

# Machine Learning & Tree Modeling


## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python and Pip](#01-python-and-pip)
+ [0.2 Libraries](#02-libraries)
+ [0.3 R and RStudio](#03-r-and-rstudio)
- [1.0 Background](#10-background)
- [2.0 Decision Trees](#20-decision-trees)
+ [2.1 Decision Trees in R](#21-decision-trees-in-r)
+ [2.2 Pruning Decision Trees](#22-pruning-decision-trees)
+ [2.3 Prediction](#23-prediction)
- [3.0 Random Forests](#30-random-forests)
+ [3.1 Algorithm](#31-algorithm)
+ [3.2 Advantages](#32-advantages)
+ [3.3 Limitations](#33-limitations)
+ [3.4 Random Forests in R](#34-random-forests-in-r)
+ [3.5 Variable Importance](#35-variable-importance)


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
