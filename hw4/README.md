# HW4
There are 3 parts

First part, it's a PCA implementation  
Second part, its a word2vec implementation  
Third part, it's a dimension fuess game(?)  

About the thrid problems, there's description below:  
>In this assignment, you have to predict the intrinsic dimension of the given sets of data points. Each set contains 10k ~ 100k data points. The data points were sampled from a Standard Normal Distribution in the intrinsic dimension (that corresponds to the particular set) and then are embedded into a higher dimension via a 3-layer neural network, with ELU activation in the first two layers and linear activation in the last layer. There are 200 sets in total.

## Kagge Link
<https://inclass.kaggle.com/c/ntu-ml2017-spring-hw4>  

## Data Link
<http://140.112.41.83:8000/data.npz>  

### requirement.txt
```
numpy
scipy
Pillow
scikit-learn
matplotlib==1.5.3
Cython
word2vec
nltk
adjustText
```

## Reading reference:
### [problem 1]
<http://stackoverflow.com/questions/17182656/how-do-i-iterate-through-the-alphabet-in-python-please>  
<https://wellecks.wordpress.com/tag/eigenfaces/>  
<http://www.scipy-lectures.org/advanced/image_processing/#opening-and-writing-to-image-files>  

<https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html>  
<https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.load.html>  

<https://docs.python.org/dev/library/argparse.html#dest>  

<https://github.com/matplotlib/matplotlib/issues/1852>  


### [problem 2]
<http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>  
<https://docs.python.org/2/library/os.path.html>  
<https://docs.python.org/3/library/functions.html#any>  
<https://github.com/Phlya/adjustText/blob/master/examples/Examples.ipynb>  
<http://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point>  
<http://stackoverflow.com/questions/14770735/changing-figure-size-with-subplots>  
<https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>  

#### NLTK
<http://www.nltk.org/data.html>  
<http://www.nltk.org/book/ch05.html>  

#### word2vec
<https://github.com/danielfrg/word2vec/blob/master/examples/word2vec.ipynb>  
<https://github.com/danielfrg/word2vec/blob/master/word2vec/scripts_interface.py>  
<https://www.quora.com/What-is-hierarchical-softmax>  
<https://arxiv.org/abs/1411.2738>  

### [problem 3]
<https://air.unimi.it/retrieve/handle/2434/247716/336817/phd_unimi_R09470.pdf>  

<https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html>  
<http://scikit-learn.org/stable/modules/neighbors.html>  
<http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.fit>  

<https://www.quora.com/Can-sklearn-algorithms-take-advantage-of-multi-core-machine>  
<https://www.quora.com/Does-scikit-learn-support-parallelism>  

<https://docs.scipy.org/doc/numpy/reference/generated/numpy.argwhere.html#numpy.argwhere>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html>  


#### np.matmul = @ v.s. np.dot
<http://stackoverflow.com/questions/6392739/what-does-the-at-symbol-do-in-python>  
<http://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication>  
<http://stackoverflow.com/questions/33982359/why-does-numpy-dot-behave-in-this-way/33982622>  
<http://stackoverflow.com/questions/41124979/how-using-dot-or-matmul-function-for-iterative-multiplication-in-python>  

#### matplotlib $DISPLAY error on work station
<http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined>  
