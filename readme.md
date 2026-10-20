# Wildlife Identification System Based on YOLOv11+Django+DeepSeek with a Login Interface in Python Source Code + ONNX Model + Beautiful Web Interface

Effect Display: Login interface main interface based on YOLO11 target detection system, supports image, video and real-time camera detection, integrated DeepSee.

## Effect Display

Login interface

![](01-83d3e475.jpg)

Main Interface

![](02-47b2a975.jpg)

A target detection system based on YOLO11, supporting image, video, and real-time camera detection, integrated with DeepSeek AI analysis functions.

### Functional Characteristics

-

Image target detection

-

Video file detection

-

Real-time camera detection

-

DeepSeek AI Intelligent Analysis

-

Adjustable detection parameters (confidence, IoU)

-

Beautiful user interface

### [Test Environment]

windows10

anaconda3+python3.10

torch 2.3.1

ultralytics 8.4.7

Django==5.2.11

### Model can detect 12 categories

['Bear', 'Buffalo', 'Camel', 'Cat', 'Cow', 'Deer', 'Dog', 'Donkey', 'Elephant', 'Leopard', 'Lion', 'Tiger']

[Training Dataset Introduction]

Dataset format: YOLO format (txt file without split path, only contains jpg images and corresponding yolo format txt files)

Number of images (number of jpg files): 5723

Number of annotations (number of txt files): 5723

Number of training set: 5013

Number of Validation Sets: 358

Test set size: 352

Label category count: 12

Label category names (Note that the order of yolo format categories does not correspond to this, but is based on the 'labels' folder with 'classes.txt' as the standard): ['Bear', 'Buffalo', 'Camel', 'Cat', 'Cow', 'Deer', 'Dog', 'Donkey', 'Elephant', 'Leopard', 'Lion', 'Tiger']

Number of boxes labeled in each category:

Bear (bear) frame count = 536

Buffalo (Water Buffalo) frame count = 664

Camel (camel) frame count = 655

Cat (cat) frame count = 484

Cow (cow) frame count = 548

Deer (deer) frame count = 603

Dog (dog) frame count = 1132

Donkey (donkey) frame count = 1165

Elephant (elephant) frame count = 447

Leopard (Leopard box = 525

Lion (lion) frame count = 539

Tiger (tiger) frame count = 588

Total frame count = 7886

Using labelImg:

Annotation rules: Draw a rectangle around the category.

Important Notice: No

Special : This dataset does not guarantee the accuracy of the trained models or weight files.

Image Preview:

![](03-c3d72468.jpg)

![](04-bef263bd.jpg)

Annotation example:

![](05-05583193.jpg)

[Training Information]

Number of training set images = 5016

Number of validation set images = 358

Training map = 90.1%

Training Accuracy = 90.4%

Training recall rate (Recall) = 83.9%

[Verification set accuracy]

Category = Map50 (%)

all=90

Bear=96

Buffalo=91

Camel=95

Cat=97

Cow=75

Deer=88

Dog=87

Dnotallow=77

Elephant=96

Leopard=100

Linotallow=80

Tiger=99

### Install Dependencies

```
pip install -r requirements.txt
```

### Database Migration

```
python manage.py makemigrations
python manage.py migrate
```

### Running Projects

```
python manage.py runserver
```

Visit http://127.0.0.1:8000

### Default login information

-

Username: admin

-

Password: admin

### Main File Structure of the Project

```
yolo11-django-deepseek-det-system/
├── manage.py               # Django management script
├── yolo_detection/ # Project configuration directory
│   ├── __init__.py
├── settings.py # Project settings
├── urls.py # Main URL configuration
└── wsgi.py # WSGI configuration
├── detection/  # Detection Application
│   ├── __init__.py
├── apps.py # Application configuration
├── forms.py
├── models.py
├── urls.py # Application URL configuration
└── views.py # View functions
├── templates/ # Template directory
├── test_img/ # Test image directory
├── index.html # Main page
└── login.html # Login Page
├── static/  # Static file directory
├── media/  # Media file directory
└── uploads/ # Upload Files Directory
├── requirements.txt  # Dependency list
├── weights/  # Template directory
├── yolov11n.onnx # Model file
└── results.png # Training log image
├── class_names # Model category file
└── model_description.txt # Brief introduction to the training and process map0.5 parameters
```

### Technology Stack

-

Django 4.2+

-

YOLO11 (Ultralytics)

-

OpenCV

-

DeepSeek API

-

Chart.js

-

HTML5/CSS3/JavaScript

### Notes

-

Ensure that the `yolov11n.onnx` model file is in the project's weights directory.

-

DeepSeek API key needs to be configured in `settings.py`

-

Recommend using Python 3.8+

### Common Evaluation Parameters Introduction

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

Class: This usually refers to the category of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

Images: Represents the number of images in the validation set. The validation set is used to evaluate model performance and is separate from the training set to ensure fairness in evaluation results. Instances: Total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images with an average of 5 target objects per image, then Instances would be 500.

P (Precision): Precision is the ratio of true positives among the instances predicted as positive by the model. The formula is: Precision = TP / (TP + FP), where TP represents True Positives and FP represents False Positives.

R (Recall): Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

mAP50: Mean Average Precision at IoU (Intersection over Union) threshold of 0.5, which represents the average precision when the IoU (Intersection over Union) is set to 0.5. The IoU metric measures the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall, used to evaluate the performance of models across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

mAP50-95: Refers to the average precision of a model within the range of IoU from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and true ones. In object detection tasks, a higher IoU threshold implies that the model needs to be more accurate in target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a comprehensive assessment of the model's performance. These metrics collectively form an important framework for evaluating the performance of object detection models. By comparing the performance of different models on these metrics, one can determine which model may be more effective in practical applications.

### Common Issues

In target detection training, a low Mean Average Precision (MAP) may be due to one of the following reasons:

Reason two: Small Objects: If the dataset contains a large number of small objects, it is likely to result in a low map value. This is because the features of small objects are not distinct, making it difficult for the model to learn them.

Reason 3: Mismatched model tuning parameters: For instance, if the learning rate is adjusted too high, it may lead to an overly rapid learning capability, resulting in disordered adjustments of model parameters.

Reason 4: Overfitting (This is now a rare occurrence in models): If the model performs exceptionally well on training data but poorly on validation or test data, it may be due to overfitting. This is often caused by an excessive number of model parameters relative to the amount of training data available, leading the model to learn noise or specific patterns from the training data and fail to generalize to new data. Nowadays, target detection models are well-equipped to handle this situation, with very few instances of overfitting occurring.

Reason 5: Different scenes: The validation set has high accuracy, but the test set does not, which may be due to inconsistencies between the training model's scene images and those in the test map. To address these issues, the following measures can be taken to improve MAP:

(1) Optimize model structure: Select the appropriate model based on the characteristics of the task and dataset, and try using different network architectures and components to improve model performance.

(2) Enhance Data Preprocessing: Proper preprocessing and enhancement of data, such as data normalization, missing value imputation, data expansion, etc., to enhance the generalization capability of the model.

(3) Adjusting the Loss Function: Try using different loss functions or combining multiple loss functions to optimize model performance.

(4) Optimize training strategies: Adjust hyperparameters such as learning rate, batch size, and number of training iterations, and employ optimization algorithms like learning rate decay and momentum to enhance model training effectiveness.

(5) Use pre-trained models: Transfer learning using pre-trained models on large datasets can accelerate model convergence and improve performance. (6) Increase the number of datasets

### Provided documents

Python source code

yolo11n.onnx model (without providing pytorch model)

Training map, P, R curve graph (in weights\results.png)

Test images (under test_img folder)

Please note that the dataset is provided in the dataset_address.xlsx file.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
