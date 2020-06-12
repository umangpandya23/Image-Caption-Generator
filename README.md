# Image-Caption-Generator

Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.

The objective of this project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM. In this Python project, I have implement the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

For the image caption generator, i have used the Flickr_8K dataset. There are also other big datasets like Flickr_30K and MSCOCO dataset but it can take weeks just to train the network so i have been using a small Flickr8k dataset. The Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”). The advantage of a huge dataset is that we can build better models.

Pre-requisities of this project: tensorflow, keras, pillow, numpy, tqdm and jupyterlab

Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images. CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc. It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.

LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

So, in this CNN-RNN based image caption model CNN is used for extracting features from the image. We will use the pre-trained model Xception and LSTM will use the information from CNN to help generate a description of the image.

The below files are developed while making the project.
Models – It will contain our trained models.
Descriptions.txt – This text file contains all image names and their captions after preprocessing.
Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
Tokenizer.p – Contains tokens mapped with an index value.
Model.png – Visual representation of dimensions of our project.
Testing_caption_generator.py – Python file for generating a caption of any image.
Training_caption_generator.ipynb – Jupyter notebook in which we train and build our image caption generator.

Here to extract feature vector from all images is done using technique called transfer learning, we don’t have to do everything on our own, we use the pre-trained model that have been already trained on large datasets and extract the features from these models and use them for our tasks. We are using the Xception model which has been trained on imagenet dataset that had 1000 different classes to classify. We can directly import this model from the keras.applications . Make sure you are connected to the internet as the weights get automatically downloaded. Since the Xception model was originally built for imagenet, we will do little changes for integrating with our model. One thing to notice is that the Xception model takes 299*299*3 image size as input. We will remove the last classification layer and get the 2048 feature vector.
model = Xception( include_top=False, pooling=’avg’ )
The function extract_features() will extract features for all images and we will map image names with their respective feature array. Then we will dump the features dictionary into a “features.p” pickle file.

In this advanced Python project, i have implemented a CNN-RNN model by building an image caption generator. Some key points to note are that our model depends on the data, so, it cannot predict the words that are out of its vocabulary. We used a small dataset consisting of 8000 images. For production-level models, we need to train on datasets larger than 100,000 images which can produce better accuracy models.