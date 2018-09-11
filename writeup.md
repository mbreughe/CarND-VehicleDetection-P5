## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[windows]: ./output_images/windows.jpg
[box_1]: ./output_images/test1_boxes.jpg
[box_2]: ./output_images/test2_boxes.jpg
[box_3]: ./output_images/test3_boxes.jpg
[box_4]: ./output_images/test4_boxes.jpg
[box_5]: ./output_images/test5_boxes.jpg
[box_6]: ./output_images/test6_boxes.jpg
[heat1]: ./output_images/test1.jpg
[heat2]: ./output_images/test2.jpg
[heat3]: ./output_images/test3.jpg
[heat4]: ./output_images/test4.jpg
[heat5]: ./output_images/test5.jpg
[heat6]: ./output_images/test6.jpg
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/mbreughe/CarND-VehicleDetection-P5/blob/master/writeup.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

In train_classifier.py at lines 55 through 66 I call the extract_features function, defined in lesson_functions.py. In lines 30 to 53 I specify the parameters used for bot HOG feature and color feature extraction. In addition I stored them in a dictionary that I later write to a pickle (see next section).

I settled for the following parameters:
* color_space = 'HLS'
* orient = 9  
* pix_per_cell = 16
* cell_per_block = 2 
* hog_channel = "ALL"
* spatial_size = (16, 16) 
* hist_bins = 32    
* spatial_feat = True 
* hist_feat = True 
* hog_feat = True 

I used HLS because it feels more natural to extract car-features independent of their actual color. I decided to bump up the number of pixels per cell to keep the feature vector low. On the other hand I chose to use all color channels as I believe this would contain more information. 



#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM with a C of 0.001 to allow for better generalization. 

The feature vector is formed by HOG features, color histogram features and spatial features (a template). The exact parameters were chosen as in the previous paragraph. Based on this, our feature vector has 1836 dimensions. This is relatively low, and was chosen intentionally to prevent overfitting and to keep the number of features much smaller than the number of training examples (aka curse of dimensionality). The parameters that increase the vector size the most are pix_per_cell and spatial_size: (64/pix_per_cell) increases HOG features quadratically, whereas spatial size increases spatial features quadratically. Increasing the amount of hist_bins on the other hand doesn't significantly increase the vector, so I chose for a relatively large value.

In addition I normalized the features using scikit's StandardScaler().

Finally I dumped the SVM, the scaler and the parameters used into a pickle, that I can reuse for usage in the rest of this project.

The classifier has an accuracy of 0.98 on the validation set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As can be seen in the image below, I scan the image only on the lower half (as the top is sky). Further, I used a variety of scales: I used windows of size 96x96 pixels to 112x112 pixels in steps of 16 pixels in both dimensions. I used a stride of half a window for each of these. In addition, near the horizon there is a strip where I used the smallest windows: 64x64. I used an overlap of 80% here. This really helped keeping track of cars that are a little further in the distance. Also, limiting the small windows to this area reduces computation time. They are not necessary towards the bottom of the screen anyway as cars will appear much larger nearby.

This code lives in search_window.py script. Around line 263 I defined a SearchWindow tuple that has spatial size, overlap and vertical positions between which to look. The run_pipeline function (line 169) creates windows out of the SearchWindow tuple and calls search_windows. In the search_windows function (line 40), features are extracted in the exact same way as we trained our SVM with. Next our SVM decides in each of the windows whether a car has been found.
 
![Sliding windows][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are examples that show which windows the SVM classifier detected to contain cars. There are a couple of false positives and one false negative. I improved the performance of the classifier by looking at the output of the SVM's decision_function, besides just using its classification result. This eliminated many false positives. As mentioned before, using multiple small windows of 64x64 with lots of overlap in a strip near the horizon significantly reduced the amount of false negatives.

![test1][box_1]
![test2][box_2]
![test3][box_3]
![test4][box_4]
![test5][box_5]
![test6][box_6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections (windows) for the last 15 frames of the video.  From these detections I created a heatmap and then thresholded that map to identify hot pixels. As a minimum 12 windows need to have covered a pixel in order for it to be hot. These 12 windows can come from 12 different frames, or some can come from the same frame. I then used `scipy.ndimage.measurements.label()` to identify individual blobs of overlapping rectangles in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Both the history of 15 frames and the threshold of 12 are parameters that can be tuned. Their current value proved to be the best in my experiments. 


### Here are six frames where the final bounding box is marked, as well as the heatmap where they originated from:

![heatmap and corresponding bounding box][heat1]
![heatmap and corresponding bounding box][heat2]
![heatmap and corresponding bounding box][heat3]
![heatmap and corresponding bounding box][heat4]
![heatmap and corresponding bounding box][heat5]
![heatmap and corresponding bounding box][heat6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


Note that in alt_implementation, I implemented the faster HOG-extraction, where data is reused for across overlapping windows. This was buggy and after spending a few hours of debugging I decided to rely on the slow method which was already giving descent results at the time. It made finetuning slow (40min per run), which was a trade-of with spending more time on debugging the fast method.

Initially I tried using a moving average approach: instead of keeping around the result of the last N frames, I chose to reduce the importance of older detections. I did this by keeping a single heatmap, which I multiplied by a number less than one (e.g., 0.2 or 0.75). This was slower and gave slightly worse results.

Because I had many false positives initially, I ended up reating more training data for the SVM. However, in a pool of so many training examples, the 20 images I extracted will probably not have contributed much as I didn't see any improvement.

I believe the SVM was only trained on car images, so it will probably fail in detecting motorcycles.

In the future I hope I can try out YOLO for the detection. 
