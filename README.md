# Deep learning–assisted literature mining for in vitro radiosensitivity data
Authors: Shuichiro Komatsu, Takahiro Oike

Provides programs for mining in vitro radiosensitivity data in scientific publications. These programs are described in our paper, "Computer-aided data mining method for in vitro radiosensitivity".

![overview_figure1](https://github.com/shuichirokomatsu/mining-for-in-vitro-radiosensitivity/blob/master/Figure1.png)

## Classifier #1 
	Detects and extract line graphs and bar graphs used in scientific publications. This program is implemented by Google Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection). 

	The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. Download TensorFlow Object Detection API repository from GitHub (https://github.com/tensorflow/models). Leave all the files in "the coarse classifier" folder into the under \object_detection folder. Make \object_detection\photo and \object_detection\photo_crop folder. Move pdf page images into the \object_detection\photo folder. Run the "object_crop.py" and you can find graphs extracted from the pdf page images in \object_detection\photo_crop folder.

## Classifier #2
	Distinguishes semi-logarithmic graphs from other graphs identified by the first program. This code requires the modules of tensorflow 1.4 and keras 2.0.1.
	Move pdf page images into the \pictures folder. Run the "VGG16_classifier.py" and you can find the class list of images.

## Classifier #3
	Detects keywords indicative of radiosensitivity experiments, i.e., "Gy", "survival fraction(s)", or "surviving fraction(s)" contained in image data. You have to prepare your own Google Cloud Vision API key. 
	Move the images extracted by above the first program into the \data folder. Change the API key variable in the "OCR.py". Run the "OCR.py" and you can find the key word count list in images.