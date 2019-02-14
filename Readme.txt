Here, character recognition of Hindi Language is performed. There are two files - train_model.py and predict_data.py.
They train on the dataset and build a model, and predict the given input image by the user respectively.
There is also a model file present which needs to be loaded in order for prediction to occur.
The library used is Keras and Tensorflow.

-------------------------------------------------------------------

The training is done using Convolutional Neural Network. The architecture is as follows:

Conv2D - 32 @ 3x3
ReLU
Con2D - 64 @ 3x3
ReLU
Max pooling - 2x2, stride 2
Conv2D - 64 @ 3x3
ReLU
Conv2D - 64 @ 3x3
ReLU
Max pooling - 2x2, stride 2
Dropout - 0.2
Flatten
Dense - 128
ReLU
Dense - 64
ReLU
Dense - 46 ( output layer )

------------------------------------------------------------------

To run the predict_data.py file, the path to file needs to be given when running the file.
Eg. python predict_data.py *file_path*
