# **Vehicle Detection**

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image1a]: ./output_images/color_hist_car.png
[image1b]: ./output_images/color_hist_noncar.png
[image1c]: ./output_images/spatial_bin1.png
[image1d]: ./output_images/spatial_bin2.png
[image2]: ./output_images/HOG_example.jpg
[image2a]: ./output_images/HOG_example_car.png
[image2b]: ./output_images/HOG_example_noncar.png
[image3a]: ./output_images/sliding_windows.png
[image3b]: ./output_images/sliding_windows_multiscale.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/heatmap_all.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_boxes.png
[image8]: ./output_images/output_video.png
[video1]: ./project_video.mp4

The project includes the following files:
* **P5.ipynb** containing the project pipeline in jupyter notebook format
* **output_images** folder including example images from each stage of the pipeline
* **README.md** as a writeup report summarizing the results and providing description of each output image
* **output_video.mp4**

---

### Data Exploration

Throughout the project we will be using images taken from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

We read-in all the `vehicle` and `non-vehicle` images using the code lines:  

```python
cars = glob.glob('./vehicles/**/*.png')
notcars = glob.glob('./non-vehicles/**/*.png')
```

**NOTE:** To maintain the same scaling of the images independently of their format (.png or .jpg) we always used cv2.cvtColor() and changed color space from BGR to RGB:
```python
# Read in car / not-car images
car_image = cv2.cvtColor(cv2.imread(cars[car_ind]), cv2.COLOR_BGR2RGB)
notcar_image = cv2.cvtColor(cv2.imread(notcars[notcar_ind]), cv2.COLOR_BGR2RGB)
```
This way we would keep consistency between the training data features and influence features.

Then, we used the auxiliary function `data_look()` to get some characteristics of the data set as seen in the following table:

| Images   | Quantity |  Size      | Data type |
|:--------:|:--------:| :----------:|:---------:|
| Cars     | 8792     |(64, 64, 3) | uint8     |
| Non-Cars | 8968     |(64, 64, 3) | uint8     |

An example of `vehicle` and `non-vehicle` class is:

![alt text][image1]

### Histograms of Color

In order to detect the objects of interest we need to extract useful features out of them, which will uniquely characterize them and separate them from all the others. The simplest feature one can get from images consists of raw color values (template matching). However, this feature is not useful for detecting things that vary in their appearance.

In feature extraction it is important to use transformations that are robust to changes in appearance. One such transform is to compute the histogram of color values in an image. Using function `color_hist()` from lecture notes we were able to extract these features and exploit them to detect a car.

Here is a visualization of the `color_hist()` function of a car image

![alt text][image1a]

Here is an example of a non-car image

![alt text][image1b]

### Spatial Binning of Color

Since the *histogram of color* features did not rely on raw pixel values (template matching), we could scale down the resolution of the images and still retain enough information to identify the vehicles. This was done using the function `bin_spatial()` from the lecture notes. Examples of this function on a car and non-car image can be seen in the figures

![alt text][image1c]

![alt text][image1d]

### Histogram of Oriented Gradients (HOG)

To handle classes of objects that vary in shape and color we need to make use of more robust representations like gradients or edges. A method of feature extraction that computes the histogram of gradient directions or orientations was introduced by Navneet Dalal and Bill Triggs and is called [Histogram of Oriented Gradients (HOG)](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

The function used to extract HOG features from our images is called `get_hog_features()`, which actually made use of the [scikit-image](http://scikit-image.org) function [`hog()`](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog). This function takes in a single color channel or grayscaled image as input, as well as various configuration parameters such as *orientations*, *pixels_per_cell* and *cells_per_block*.

Visualization of the `get_hog_features()` function applied on a car and a non-car test images (transformed to YUV color space) using HOG parameters:

```python
# HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

can be seen in the following figures

![alt text][image2a]

![alt text][image2b]

#### HOG parameters selection

In order to decide which HOG and Color feature parameters to use, we instinctively tried various combinations and settled to the ones with the best test accuracy and prediction time. Some of the most prominent of the results can be seen in the following table:

| Parameters      |  Set 1  | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Set 7 |
|:--------------:|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| color_space    | 'YUV'   | 'YUV' | 'YUV' | 'YUV' | 'YUV' | 'YUV' | 'YUV' |
| orient         | 11      | 9     |  7    |   7   |   11  |   9   |   13  |
| pix_per_cell   | 16      | 8     |  16   |   16  |   8   |  16   |   16  |
| cell_per_block | 2       | 2     |  2    |   2   |   2   |   2   |   2   |
| hog_channel    | 'ALL'   | 'ALL' | 'ALL' | 'ALL' | 'ALL' | 'ALL' | 'ALL' |
| spatial_size   | (16,16) |(16,16)|(16,16)|(16,16)|(16,16)|(16,16)|(16,16)|
| hist_bins      | 32      | 16    |  16   |  32   |  16   |  32   |  32   |
| **Training Results**| **Set 1**|**Set 2** | **Set 3** | **Set 4** | **Set 5** | **Set 6**| **Set 7**|
| Test Accuracy  | 0.9876  |0.987  |0.9876 |0.9885 |0.9882 | 0.99  | 0.99  |
| Prediction Time (ms)|  2.71   | 3.6   | 2.86  | 2.82  | 5.52  | 2.98  | 2.86  |
| Number of Features   |  2052   | 6108  | 1572  | 1620  | 7284  | 1836  | 2268  |

We included the *Prediction Time* in the optimization criteria since it is really important for the classifier to "work" fast in order to make real-time predictions. We finally chose **Set 7** to train our classifier, which is a good compromise between test accuracy and prediction speed.

### Train a classifier using HOG and Color features

#### Extract HOG and color features

To extract the HOG and Color features we used the function `extract_features()` based on the lecture notes with the following parameters (**Set 7**):

| Parameter  | Value | Description |
|:--------:|:--------:| :--------:|
| color_space  | 'YUV'   | Colorspace (RGB, HSV, LUV, HLS, YUV, YCrCb) |
| orient | 13     | HOG orientations |
| pix_per_cell   | 16  | HOG pixels per cell |
| cell_per_block   | 2  | HOG cells per block |
| hog_channel   | 'ALL'   | HOG channel (0, 1, 2 or 'ALL') |
| spatial_size   | (16,16)  | Spatial binning dimensions |
| hist_bins   | 32   | Number of histogram bins |

#### Train a linear SVM classifier

In order to train a classifier we initially need to normalize our data. To do so we used the `StandardScaler()` method from Python's [sklearn package](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) into the array stack of feature vectors `X = np.vstack((car_features, notcar_features)).astype(np.float64)`.

Then we defined the labels vector with 1 for car images and 0 for non-car images, and split the normalized feature vector `scaled_X` into a training (80%) and validation set (20%) using the [sklearn package](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) `train_test_split()`.

Finally we trained a linear SVM using `LinearSVC()` and tested its prediction accuracy using the method `svc.score()`.

The results obtained for 10 prediction labels were:

```python
5.86 Seconds to train SVC...
Test Accuracy of SVC =  0.9921
My SVC predicts:  [ 0.  1.  0.  0.  1.  1.  0.  1.  1.  0.]
For these 10 labels:  [ 0.  1.  0.  0.  1.  1.  0.  1.  1.  0.]
0.00286 Seconds to predict 10 labels with SVC
```


### Sliding Window Search

After having extracted all useful features from the images and having trained the linear SVC classifier using the labeled data, we had to implement a method to search for our desired objects (vehicles) in an image. A most sufficient way of doing this was to use the **HOG Sub-sampling Window Search** that only needed to extract the Hog features once. This was done using the function `find_cars()` (based on lecture notes), which returned the identified rectangles and plotted them on an output image `draw_img`.
We initially used the `find_cars()` function with the following parameters:

| Parameter  | Value | Description |
|:--------:|:--------:| :--------:|
| `x_start`  |  0 |  start of the search area (x-axis) |
| `ystart`   | 414  |  start of the search area (y-axis) |
| `ystop`   | 678  |  stop of the search area (y-axis) |
| `scale`   |  1.0 | scaling of the search window  |

Notice that we adapted the `find_cars()` function to include a starting point for the search in x-axis as well (`x_start`). The search area in x-axis would terminate at the end of the figure, therefore we needed not define a `x_stop` point.

To visualize the result, we used the images found on the [test_images](https://github.com/udacity/CarND-Vehicle-Detection/tree/master/test_images) folder of the [Project Repository](https://github.com/udacity/CarND-Vehicle-Detection), as follows:

![alt text][image3a]

#### Multiple scale window search

To increase robustness of our identification we used multiple window scales and restricted the search to the only areas of the image where vehicles might appear. We carefully chose scaling to be larger on the bottom of the image (vehicles closer to the camera) and smaller around the center of the image (vehicles near the horizon). Moreover, in order to filter vehicles appearing beyond the barrier on the left side of the road, we also included a starting point on the x-axis, which takes different values according to whether we are close to the camera or near the horizon.

These search features were implemented in the function `find_cars_multiscale()`. The scalings and search areas were defined as follows:

```python
scales = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0]

x_start_list = [450, 450 , #  for scale 1.0
                450, 450 , #  for scale 1.5
                250, 250 , #  for scale 2.0
                250, 250 , #  for scale 2.5
                150, 150]  #  for scale 3.0


y_start_stop_list = [[414, 478], [446, 510], #  for scale 1.0
                     [398, 494], [446, 542], #  for scale 1.5
                     [414, 542], [478, 606], #  for scale 2.0
                     [385, 545], [465, 625], #  for scale 2.5
                     [372, 564], [468, 660]] #  for scale 3.0
```

Visualization on the test images can be seen in the following figure:

![alt text][image3b]

### Filter Multiple Detections & False Positives

In the above examples we noticed that the classifier in some cases reported multiple overlapping instances of the same car (duplicates) or reported cars where there were none (false positives). In order to filter out these cases we created a heat-map using function `add_heat()` from the lecture notes. Then we imposed a threshold to keep only the "hot" parts where the vehicles probably were, using the function `apply_threshold()`.

Once we obtained a thresholded heat-map, in order to figure out how many cars we had in each frame and which pixels belonged to which cars, we used the `label()` function from [scipy.ndimage.measurements](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.measurements.label.html). To draw the boxes around the labeled regions we used the function `draw_labeled_bboxes()`.

All of the above steps were summarized in the function `remove_false_boxes()` that employed the functions `add_heat()`, `apply_threshold()` and `draw_labeled_bboxes()`     to remove multiple detections and false positives and returned all intermediate images.

Visualization of this function on the test images for Heatmap `thresshold = 2` can be seen in the following figures:

![alt text][image5]

### Final Pipeline ( function `Final_Pipeline()`)

Once we have extracted the features and trained our classifier, we summarized the vehicle detection algorithm in the final pipeline as follows:

1. Find cars using multiple scale window search (function `find_cars_multiscale()`)
2. Filter multiple detections and false positives (function `remove_false_boxes()`)

We have tested the pipeline on the set of test-images using the parameter configuration:

| Parameter  | Value | Description |
|:--------:|:--------:| :--------:|
| `color_space`  | 'YUV'   | Colorspace (RGB, HSV, LUV, HLS, YUV, YCrCb) |
| `spatial_feat`   | `True`  | Spatial features On or Off  |
| `hist_feat`   |  `True` | Histogram features On or Off  |
| `hog_feat`   | `True`   | HOG features On of Off  |
| `orient` | 13     | HOG orientations |
| `pix_per_cell`   | 16  | HOG pixels per cell |
| `cell_per_block `  | 2  | HOG cells per block |
| `hog_channel`   | 'ALL'   | HOG (0, 1, 2 or 'ALL') |
| `spatial_size`   | (16,16)  | Spatial binning dimensions |
| `hist_bins`   | 32   | Number of histogram bins |

and the search areas:

```python
scales = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0]

x_start_list = [450, 450 , #  for scale 1.0
                450, 450 , #  for scale 1.5
                250, 250 , #  for scale 2.0
                250, 250 , #  for scale 2.5
                150, 150]  #  for scale 3.0


y_start_stop_list = [[414, 478], [446, 510], #  for scale 1.0
                     [398, 494], [446, 542], #  for scale 1.5
                     [414, 542], [478, 606], #  for scale 2.0
                     [385, 545], [465, 625], #  for scale 2.5
                     [372, 564], [468, 660]] #  for scale 3.0
```

#### Final pipeline tested on images

![alt text][image7]


#### Final pipeline tested on video

Finally the pipeline was successfully tested on the project video. A link of the output video can be found [here](./output_video.mp4).

![alt text][image8]


### Problems/Issues

During the implementation of the project, the following problems/issues were encountered:

* In the output video one can notice some false positives appearing suddenly in shady areas or lane lines. This proves that our classifier did not work efficiently in these cases. Such problems could be resolved with a more advanced training or classification method.

* The boxes of identified vehicles seem to be unstable (trembling) between frames. Such a problem could be resolved by creating a stack variable and storing previous detections. A weighted averaging over `n` past measurements would smoothen the result and produce a more stable solution.

* In order to keep track of the previous frame measurements in a more robust way, one could use a python class.

* Other possible issues could appear in videos containing more challenging driving conditions such as steep hills, heavy traffic or adverse weather conditions (e.g. rain/snow/fog). In order to cope with such cases, one should optimize the classifier by augmenting the training data set, including figures that would correspond to such harsh environmental conditions.

### Conclusion  

In this project we have detected and tracked vehicles in an image and subsequently
in a video. To do so we have extracted Color and Histogram of Oriented Gradient (HOG) features, used them to train an SVM classifier and employed this classifier in an image to identify vehicles, performing a multiple scale sliding window search. Finally we filtered out any multiple vehicle detections and false positives. Testing of the pipeline on both images and a video successfully demonstrated the effectiveness of the approach.
