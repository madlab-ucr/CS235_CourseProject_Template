# CS235 Course Project Starter

# CS235: A generic project submission template

A bare-bones project template for the class project. 

1. **src/dataloader.py** - Loads and preprocesses your _raw data_ and converts it into a _dataset_ consiting of train-validation-test sets. Returns said train-validation-test sets.
2. **src/models.py** - Builds your models. Each model is a python class with _at least_ train() and predict() methods along with class attributes for model hyperparameters. Returns the class object.
3. **src/driver.py** - Entry-point script which calls the _dataloader_ to get train-val-test sets or generators and _models_ to get a model object to be trained using the train set.
4. **src/utils.py** - Contains utility functions that support your main scripts (plotting, etc.)

## FAQs:
<b> Can I use Scikit-Learn for ... ? </b>

_For using ML models out-of-the-box? **No**. You are supposed to code them from scratch*._ 

_For dataset preprocessing? **Yes**. You can use built-in methods for splitting your dataset into train-test sets, for dimensionality reduction (e.g. using PCA) as a preprocessing step, for visualizing your high-dimensional datasets in a 2-D/3-D plots (e.g. using t-SNE, UMAP, etc) during exploratory data analysis (EDA)._ 

_For comparing your implementation against a baseline? **Yes**. You can use the scikit learn implementation as a baseline to compare your implementation against for correctness, speed, etc. 

<b> Can I use Tensorflow Keras / PyTorch for ... ? </b>

_You are not expected to code every single neuron from scratch, so **YES**. You can use popular deep learning frameworks to - 1. design your custom neural networks, 2.  use some existing, pretrained networks for transfer learning or as a baseline._

<b> Can I customize this template to suite my project? </b>

_Absolutely.
The minimum modules required are - dataloader.py, models.py, driver.py. You could additionally add utils.py, eda.ipynb_

_(Note: Following this template will make your and the grader's life easy.)_