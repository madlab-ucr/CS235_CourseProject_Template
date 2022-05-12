# CS235 Course Project Starter

A bare-bones project template for course project submission. 
0. **EDA.ipynb** - A notebook detailing exploratory data analysis. It is crucial to perform EDA before embarking on any machine learning journey. 
1. **src/dataloader.py** - Loads and preprocesses your _raw data_ and converts it into a _dataset_ consiting of train-validation-test sets. Returns said train-validation-test sets.
2. **src/models.py** - Builds your models. Each model is a python class with _at least_ train() and predict() methods along with class attributes for model hyperparameters. Returns the class object.
3. **src/driver.py** - Entry-point script which calls the _dataloader_ to get train-val-test sets or generators and _models_ to get a model object to be trained using the train set.
4. **src/utils.py** - Contains utility functions that support your main scripts (plotting, etc.)

## FAQs:
<b> Can I use Scikit-Learn for ... ? </b>

_For dataset preprocessing? **Yes**. You can use built-in methods for splitting your dataset into train-test sets, for dimensionality reduction (e.g. using PCA) as a preprocessing step, for visualizing your high-dimensional datasets in a 2-D/3-D plots (e.g. using t-SNE, UMAP, etc) during exploratory data analysis (EDA)._ 

_For comparing YOUR implementation against a baseline? **Yes**. You can use the scikit-learn implementation as a baseline to compare your implementation against (for correctness, speed, etc.)_

<b> Can I use Tensorflow Keras / PyTorch for deep learning? </b>

_You are not expected to code every single neuron from scratch, so **YES**. You can use popular deep learning frameworks to - 1. design your custom neural networks and build your data generators, 2.  use some existing, pretrained networks for transfer learning or as a baseline to compare your custom network against, 3. use built-in optimizers (like Adam, SGD, RMSProp, etc) for back propagation, 4. use tensorboard to monitor your training._

<b> Can I customize this template to suit my project? </b>

_Absolutely, **yes**. The minimum modules required are - dataloader.py, models.py, driver.py. You could additionally have utils.py, eda.ipynb as shown in this repo. And you can have more scripts as needed by your project._

_(Note: Following this template as a group instead of dumping code in a collection of uncommented jupyter notebooks will make the grader's (and consequently your) life easier. Moreover, this repo demonstrates basic best practices for an end-to-end ML project.)_