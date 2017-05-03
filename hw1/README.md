# HW1
> 本次作業的資料是從中央氣象局網站下載的真實觀測資料，希望大家利用linear regression或其他方法預測PM2.5的數值。

> 作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。

> train.csv：每個月前20天的完整資料。

> test_X.csv：從剩下的10天資料中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

## Kaggle Link
<https://inclass.kaggle.com/c/ml2017-hw1-pm2-5/>

## Data Link
<https://drive.google.com/file/d/0B8Si647wj9ZoelM4RktXZU1BbkE/view?usp=sharing>


## Reference

<http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary>  

<http://stackoverflow.com/questions/1303347/getting-a-map-to-return-a-list-in-python-3-x>  

### reading reference
<http://aimotion.blogspot.tw/2011/10/machine-learning-with-python-linear.html>  

### numpy
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html>  
<https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.vsplit.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html>  
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html>  

### np.dot v.s np.matmul
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>  
<https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.matmul.html>  
<http://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication>  

<http://cpmarkchang.logdown.com/posts/275500-optimization-method-adagrad>  

### featuer scaling
<https://en.wikipedia.org/wiki/Feature_scaling>  
<https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling>  
<http://sebastianraschka.com/Articles/2014_about_feature_scaling.html>  

### least squares solution
<http://math.mit.edu/~gs/linearalgebra/ila0403.pdf>  



