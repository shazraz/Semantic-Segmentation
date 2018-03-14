# Semantic Segmentation

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Run
Launch your console and execute:
```
python main.py -d <DATA PATH> -e <EPOCHS> -b <BATCH SIZE>
```
The images in the runs folder were inferred using a model trained for 10 epochs on a batch size of 4. The remaining hyper-parameters are listed in the parameters text file in the runs folder.