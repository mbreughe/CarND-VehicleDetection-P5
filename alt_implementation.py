import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *

# Define a single function that can extract features using hog sub-sampling and make predictions

def alt_search_window(img, ystart, ystop, scale, svc, X_scaler, color_space = 'RGB', orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), 
    hist_bins=32, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img)     
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog = []
    hog.append(get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False))
    hog.append(get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False))
    hog.append(get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False))
    
    windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
        
            img_features = []
            
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            
            if hog_feat:
                hog_features = []
                if hog_channel == "ALL":
                    hog_features.extend( hog[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() )
                    hog_features.extend( hog[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() )
                    hog_features.extend( hog[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() )
                else:
                    hog_features = hog[hog_channel]
                    
                img_features.append(hog_features)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                img_features.append(spatial_feat)
                
            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
                img_features.append(hist_features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            #print(test_prediction)
            
            
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                windows.append(((xbox_left, ystart + ytop_draw), (xbox_left + win_draw, ytop_draw+win_draw+ystart)))
                
    return windows
    