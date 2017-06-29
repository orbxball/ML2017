# HW3
**Image Sentiment Classification**

> 本次作業為Image Sentiment Classification。我們提供給各位的training dataset為兩萬八千張左右48x48 pixel的圖片，以及每一張圖片的表情label（注意：每張圖片都會唯一屬於一種表情）。總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。

> Testing data則是七千張左右48x48的圖片，希望各位同學能利用training dataset訓練一個CNN model，預測出每張圖片的表情label（同樣地，為0~6中的某一個）並存在csv檔中。

> 相關格式及報告說明請詳閱：  
> [PPT](https://docs.google.com/presentation/d/1QFK4-inv2QJ9UhuiUtespP4nC5ZqfBjd_jP2O41fpTc/edit#slide=id.p)  
> [作業網址](https://sunprinces.github.io/ML-Assignment3/index.html)  

> [注意] 這次作業希望大家在衝高Kaggle上Accuracy的同時，對訓練的model及預測的結果多做一些觀察（P3-P5），並在報告中多加詳述。


## Kaggle Link
<https://inclass.kaggle.com/c/ml2017-hw3>

## Data Link
<https://drive.google.com/file/d/0B8Si647wj9ZoTHlJR1pDazUxSVE/view?usp=sharing>  

## Reference
<https://keras.io/#getting-started-30-seconds-to-keras>
<https://keras.io/getting-started/sequential-model-guide/>
<https://keras.io/layers/convolutional/#conv2d>
<https://keras.io/losses/>
<https://keras.io/optimizers/>
<https://keras.io/models/sequential/>
<https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model>
<https://keras.io/layers/core/>
<https://keras.io/layers/advanced-activations/>
<https://keras.io/models/model/>
<https://keras.io/visualization/#model-visualization>
<https://keras.io/callbacks/>

<https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html>
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.squeeze.html>
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.argwhere.html>
<https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html>

### Code Reference
<https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py>
<https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>
<http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t>

<https://en.wikipedia.org/wiki/Test_set>
<https://www.zhihu.com/question/23437871>
<http://scikit-learn.org/stable/index.html>
<https://en.wikipedia.org/wiki/Confusion_matrix>
<http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>

<http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/>
<http://blog.fastforwardlabs.com/2016/02/24/hello-world-in-keras-or-scikit-learn-versus.html>

### Saliency map
<https://github.com/raghakot/keras-vis/tree/master/vis>

### Visualize filters & outputs
<https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html>
<https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py>
<https://en.wikipedia.org/wiki/Median_filter>
<https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.median_filter.html>

### Papers
<http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf>
<https://arxiv.org/pdf/1608.02833.pdf>
