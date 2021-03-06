from lesson_functions import *
import pickle
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from alt_implementation import alt_search_window
from collections import deque


def reduce_heat(heatmap):
    #heatmap = np.maximum(np.subtract(heatmap, 1),0) 
    heatmap = 0.2 * heatmap
    return heatmap
    
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
    
def create_heatmap(image, bbox_list, threshold):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, bbox_list)
    return heat

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True, conf_thresh = 0.1):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        confidence = clf.decision_function(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1 and confidence > conf_thresh:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
    
def draw_labeled_bboxes(img, labels, color=(0,0,1)):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img
    
def process_image(image):
    global g_heat_q
    global g_parameters
    global g_s_winds
    global g_alt_search
    
    
    # 8 in traditional approach
    img, g_heat_q = run_pipeline(image, g_parameters, g_s_winds, heat_q=g_heat_q, threshold=12, visualize=False, alt_search=g_alt_search, frame_hist_len=15)
    
    # Don't forget to multiply by 255 again!
    return 255*img
    
def process_video(video_fname, parameters, s_winds, alt_search=False):
    # Need to define globals as VideoFileClip.fl_image only accepts one parameter
    global g_heat_q
    global g_parameters
    global g_s_winds
    global g_alt_search
    g_heat_q = None
    
    
    g_parameters = parameters
    g_s_winds = s_winds
    g_alt_search = alt_search
    
    ofname = "result.mp4"
    if alt_search:
        ofname = "result_alt.mp4"

    clip = VideoFileClip(video_fname)
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(ofname, audio=False)

def run_pipeline(img, parameters, s_winds, heat_q=None, threshold=1, visualize=True, alt_search = False, frame_hist_len=10):
    # Classifier was trained on png images. Conversion needed if we are running on jpg images
    img = img.astype(np.float32)/255
        
    detections = []
    
    if alt_search:
        for s_wind in s_winds:
            overlap_steps = int(-8 * s_wind.xy_overlap + 8)
            scale = s_wind.xy_window / 64
            detections.extend(alt_search_window(img, s_wind.y_start_stop[0], s_wind.y_start_stop[1], svc, X_scaler, **parameters, overlap_cells_per_step = overlap_steps, scale=scale))
    
    else:
        for s_wind in s_winds:

            windows = slide_window(img, x_start_stop=[None, None], y_start_stop=s_wind.y_start_stop, 
                            xy_window=(s_wind.xy_window, s_wind.xy_window), xy_overlap=(s_wind.xy_overlap, s_wind.xy_overlap))
            new_detections = search_windows(img, windows, svc, X_scaler, **parameters, conf_thresh=0.85)
            detections.extend(new_detections)
    if heat_q is None:
        heat_q = deque(maxlen=frame_hist_len)
    
    cur_heat = np.zeros_like(img[:,:,0]).astype(np.float)   
    cur_heat = add_heat(cur_heat, detections)
    
    heat_q.append(cur_heat)
    
    tot_heat = sum(heat_q)
    
    tot_heat = apply_threshold(tot_heat, threshold)
    
    heatmap = np.clip(tot_heat, 0, 255)    

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    if visualize:
        return draw_img, heat_q, heatmap
    else:
        return draw_img, heat_q


def test_pipeline(indir, outdir, parameters, s_winds, visualize=True):
    for fname in os.listdir(indir):
        if not fname.endswith("jpg"):
            continue
        
        img = mpimg.imread(os.path.join(indir, fname))
        
        ofname = os.path.join(outdir, fname)
     
        if visualize:
            (draw_img, _, heatmap) = run_pipeline(img, parameters, s_winds, threshold=1, visualize=visualize)
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
        else:
            draw_img, _ = run_pipeline(img, parameters, s_winds, threshold=1, visualize=visualize)
            plt.imshow(draw_img)
            
        plt.savefig(ofname)

def dumpVideoAtRanges(video_fname, sec_ranges, odir):
    if not os.path.exists(odir):
        os.makedirs(odir)
        
    set_no = 0
    for (low, high) in sec_ranges:
        clip = VideoFileClip(video_fname)
        clip = clip.cutout(high, clip.duration)
        clip = clip.cutout(0, low)
        fname = os.path.join(odir, "frame_s_{}_%03d.jpg".format(set_no))
        clip.write_images_sequence(fname)
        set_no += 1
    
if __name__ == "__main__":
    
    outdir = "output_images"
    indir = "test_images"
    video_fname = "project_video.mp4"

    # Load the classifier parameters
    with open("model.p", "rb") as ifh:
        parameters = pickle.load(ifh)
        X_scaler = pickle.load(ifh)
        svc = pickle.load(ifh)
    
    # Set search window parameters
    SearchWindow = namedtuple('SearchWindow', ['y_start_stop', 'xy_window', 'xy_overlap'])   
    s_winds = []
    s_winds.append(SearchWindow([400, 500], 64, 0.8))
    
    for i in range(96, 128, 16):
        s_winds.append(SearchWindow([350, 670], i, 0.5))
    
    #test_pipeline("debugging_final", outdir, parameters, s_winds, True)
    process_video(video_fname, parameters, s_winds, alt_search=False)

    #dumpVideoAtRanges(video_fname, [(25,26)], "debugging_final")
        
        