import os
import cv2
import tqdm
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import io, morphology


def dir_score(wsi_masks_folder_path, wsi_path, fov_x, fov_y, overlap_x, overlap_y, smp_size, save_fov):
    """
    Calculate the MLI (Mean Linear Intercept) score for Whole Slide Images (WSI).

    :param wsi_masks_folder_path: (str): Path to the folder where the WSI masks are located.
    :param fov_x: (int): FOV size in the x direction.
    :param fov_y: (int): FOV size in the y direction.
    :param overlap_x: (float): Overlap percentage in the x direction (between 0 and 0.99).
    :param overlap_y: (float): Overlap percentage in the y direction (between 0 and 0.99).
    :param smp_size: (int): Size of the guideline to superimpose.
    :param wsi_path: (str): Name of the original WSI.
    :param save_fov: (int): Flag indicating whether to save FOVs or not (0 - do not save, 1 - save).
    :return: None

    """

    wsi_mask_list = ['BG.tif', 'BR.tif', 'BV.tif', 'AVLW.tif']
    wsi_name, wsi_format = os.path.splitext(os.path.basename(wsi_path))
    wsi_parent_path = os.path.dirname(wsi_path)

    # load the respective WSI masks
    BG_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[0]), 0)

    # Ensure the user input values are not invalid
    if fov_x > BG_BW_WSI.shape[0]:
        raise ValueError('Error: fov_x cannot be greater than the image x length.')

    if fov_y > BG_BW_WSI.shape[1]:
        raise ValueError('Error: fov_y cannot be greater than the image y length.')

    if overlap_x > 0.99 or overlap_y > 0.99 or overlap_x < 0 or overlap_y < 0:
        raise ValueError('Error: overlap cannot be greater than 0.99 (99%) or less than 0 (no overlap).')

    if save_fov < 0 or save_fov > 1:
        raise ValueError('Error: save_fov must be either 0 (do not save FOVs) or 1 (save FOVs).')

    print("Starting MLI Calculation using Direct Method")
    BR_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[1]), 0)
    BV_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[2]), 0)
    AVLW_BW_WSI=cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[3]), 0)

    mli_folder = os.path.join(wsi_parent_path, f"{wsi_name}_MLI")        
    os.makedirs(mli_folder, exist_ok=True)

    if save_fov == 1:
        # load original WSI and split into rgb channels
        ORIGINAL_WSI = io.imread(wsi_path)
        ORIGINAL_WSI = ORIGINAL_WSI[:, :, :3]  # remove Alpha channel
        # split into RGB color channels
        r_ORIGINAL_WSI, g_ORIGINAL_WSI, b_ORIGINAL_WSI = cv2.split(ORIGINAL_WSI)

        # create our file to store FOVs with guidelines superimposed
        guideline_fov_save_location = wsi_parent_path + '/' + wsi_name + '_MLI' + '/' + wsi_name+ '_FOV_Guideline_Direct_MLI'
        os.makedirs(guideline_fov_save_location, exist_ok=True)

 #count used for FOV image numbering (ex. img_1, img_2...etc)
    count = 1 
    #loc_x1 and loc_x2 used for tracking horizontal pixel location of FOV with respect to WSI 
    loc_x1 = 1
    loc_x2 = fov_x
    #loc_y1 and loc_y2 used for tracking vertical pixel location of FOV with respect to WSI 
    loc_y1 = 1
    loc_y2 = fov_y
    #shift_x and shift_y determines change in horizontal/vertical pixel location for each FOV, respectively
    #ex. if overlap_x is 30% overlap, then the next FOV image in x direction must be shifted 70%. shift_x = # pixels to shift
    shift_x = fov_x*(1-overlap_x)
    shift_y = fov_y*(1-overlap_y)

    #create dataframe and lists to track information
    original_naming = [wsi_name]
    df = pd.DataFrame(original_naming, columns = ['Original Image Name'])
    FOV_name_list = []#track FOV image number
    pixel_loc_x1_list = []#track x1 pixel location
    pixel_loc_x2_list = []#track x2 pixel location
    pixel_loc_y1_list = []#track y1 pixel location
    pixel_loc_y2_list = []#track y2 pixel location
    acc_rej_list = [] #initialize a list to store whether the FOV was rejected or accepted
    #if the image is ACC:
    crossing_list = [] #create a list to track the # crossings per FOV
    crossing_total_count = 0 #keep track of overal # crossings for entire FOV set
    acc_fov_count = 0 #keep track of number of accepted FOVs
    chord_length_list = [] #keep track of each individual chord length measurement
    total_wsi_chord_length = 0 #track the sum of all the chords measured
    total_wsi_chords_measured = 0 #track the number of chords measured in the entire WSI
    chord_image_name_list = [] #keep track which chord measurements belong to which image
    
    #extract the FOVs based on FOV size and overlap
    for i in tqdm.tqdm(range(0, BG_BW_WSI.shape[0], int(fov_y * (1 - overlap_y))), desc="Processing FOVs"):
        for j in range(0, BG_BW_WSI.shape[1], int(fov_x*(1-overlap_x))):  #Steps of fov_x with overlap_x
            #grab FOV for each of the 4 masks
            single_patch_BG = BG_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_BR = BR_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_BV = BV_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_AVLW = AVLW_BW_WSI[i:i+fov_y, j:j+fov_x]
            
            if save_fov == 1:
                #grab each color channel FOV from the original wsi 
                r_patch_original = r_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]
                g_patch_original = g_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]
                b_patch_original = b_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]

            # only keep FOVs that are same shape as fov_x,fov_y, remove smaller patches around the sides
            if single_patch_BG.shape[0] == fov_y and single_patch_BG.shape[1] == fov_x:
                #track image number, x and y pixel locations (dataframe)
                FOV_name_list.append('img_'+str(count))
                pixel_loc_x1_list.append(int(loc_x1))
                pixel_loc_x2_list.append(int(loc_x2))
                pixel_loc_y1_list.append(int(loc_y1))
                pixel_loc_y2_list.append(int(loc_y2))            

                #now superimpose guidline onto FOV masks (BG, BR, BV, AVLW) to either accept or reject the specific FOV location            
                #first convert FOVs into datatype bool
                fov_bg = np.array(single_patch_BG, dtype=bool)
                fov_br = np.array(single_patch_BR, dtype=bool)
                fov_bv = np.array(single_patch_BV, dtype=bool)
                #look at every pixel that composes the guidline and determine if there is a "True" value, and reject if "True":
                rej_trigger = 0 #if the item is rejected, no need to determine # of crossings
                #check if guidline touches the background
                for pixel_num in range (0,smp_size):
                    if fov_bg[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                        rej_trigger = 1
                        acc_rej_list.append('REJ_BG')
                        break
                #check if guideline touches a bronchi
                if rej_trigger == 0: #if the FOV has not been rejected because of background
                    for pixel_num in range (0,smp_size):
                        if fov_br[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                            rej_trigger = 1
                            acc_rej_list.append('REJ_BR')
                            break
                #check if guidline touches a blood vessel
                if rej_trigger == 0: #if the FOV has not been rejected because of background and BR
                    for pixel_num in range (0,smp_size):
                        if fov_bv[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                            rej_trigger = 1
                            acc_rej_list.append('REJ_BV')
                            break

                #if we reach this point and rej_trigger is still 0, the image is accepted and we can calculate # crossings
                if rej_trigger == 0:
                    acc_rej_list.append('ACC')
                    acc_fov_count = acc_fov_count + 1 #update our count for the number of accepted FOVs
                    cross_1 = 0 #keep track of transition from False to True (avleolar/ductile space into septa wall)
                    cross_2 = 0 #keep track of transition from True to False (septa wall into avleolar/ductile space)
                    crossings = 0 #keep track of number of crossings that occured in the FOV
                    pixel_chord_length = 0 #number of pixels representing a chord
                    first_cross = 0 #track when the first complete crossing occurs in the image
                    first_cross_loc = 0 #record location where first crossing occurs in an the FOV (left side)
                    last_cross_loc = 0 #record location where last crossing occurs in an FOV (left side)
                    
                    if save_fov == 1:
                        #used these arrays to temporarily store the original colors where the guideline will be superimposed 
                        store_r_channel_accept = []
                        store_g_channel_accept = []
                        store_b_channel_accept = []
                        r_accepted_fov = r_patch_original
                        g_accepted_fov = g_patch_original
                        b_accepted_fov = b_patch_original

                        #superimpose a blue guideline and then change to red for the areas that have a crossing later
                        #include the entire width to store and restore
                        for pixel_num_5 in range(0,fov_x):
                            store_r_channel_accept.append(r_accepted_fov[round(fov_y/2), pixel_num_5])
                            store_g_channel_accept.append(g_accepted_fov[round(fov_y/2), pixel_num_5])
                            store_b_channel_accept.append(b_accepted_fov[round(fov_y/2), pixel_num_5])
                        for pixel_num_6 in range(0,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 255
                    
                    #perform post processing on the AVLW image to remove noise:
                    #convert to datatype bool
                    fov_avlw = np.array(single_patch_AVLW, dtype=bool)
                    #remove small white objects from alveoli space
                    image_avlw_bwareopen = morphology.remove_small_objects(fov_avlw, 250)#pixel limit trial/error
                    #NOT (~) binary image
                    image_avlw_not = ~image_avlw_bwareopen
                    #remove small black objects from septa walls
                    image_avlw_not_bwareaopen = morphology.remove_small_objects(image_avlw_not, 500)#pixel limit trial/error
                    #convert back to original binary image with black/white small objects removed
                    image_avlw = ~image_avlw_not_bwareaopen 

                    #iterate through the superimposed guidline to determine crossings
                    for pixel_num in range (0,smp_size):
                        #if transition from pixel_num to pixel_num + 1 is from alveolar/ductile space to septa wall,
                        #and this is the first crossing found, update cross_1:
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == True and first_cross == 0:
                            cross_1 = 1
                            first_cross_loc = pixel_num #store location where first crossing begins
                            
                        #if transition from pixel_num to pixel_num + 1 is from alveolar/ductile space to septa wall,
                        #and this is for further crossings, we can now start to RECORD chord lengths
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == True and first_cross == 1:
                            cross_1 = 1
                            last_cross_loc = pixel_num
                            if pixel_chord_length > 1:
                                chord_image_name_list.append('img_'+str(count))
                                chord_length_list.append(int(pixel_chord_length)*0.497885)#convert from pixels to micrometers and record
                                total_wsi_chord_length = total_wsi_chord_length + (int(pixel_chord_length)*0.497885)
                                total_wsi_chords_measured = total_wsi_chords_measured + 1
                            
                        ##if cross_1=1 and cross_2=0, we can now make this area of the guideline red
                        if cross_1 == 1 and cross_2 == 0:
                            pixel_chord_length = 0 #we restart count for pixel_chord_length
                            if save_fov == 1:
                                r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 255
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                                #no need to change green channel, it has already been set to 0 before 
                            

                        #if transition from pixel_num to pixel_num + 1 is from septa wall into alveolar/ductile space
                        #and we already crossed into septa region from avleolar/ductile space previously, update cross_2:
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == False and cross_1 == 1:
                            cross_2 = 1
                            first_cross = 1                          
                        
                        #if we have made a full transition through a septa wall (cross_1 = 1 and cross_2 = 1), crossings++
                        if cross_1 == 1 and cross_2 == 1: #fully crossed a septa wall
                            crossings = crossings + 1
                            cross_1 = 0 #reset to find next crossing
                            cross_2 = 0 #reset to find next crossing
                            last_crossing_loc = pixel_num #location of right-side of last crossing 
                        
                        #if we have made a full crossing then we can increase our pixel chord length and make the chord green
                        if cross_1 == 0 and cross_2 == 0 and first_cross == 1:
                            pixel_chord_length = pixel_chord_length + 1 
                            if save_fov == 1: #update FOV guideline if the user requests FOVs to be generated                           
                                g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 255
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                                
                        #if we reach the last pixel and we are inside alveolar/ductile space, keep going until we reach a septa wall
                        #if we have not made any crossings, ignore as this is taken care of at a further step
                        if cross_1 == 0 and cross_2 == 0 and pixel_num == smp_size - 1 and first_cross == 1:
                            finish_final_chord = 0
                            reverse_loc = pixel_num #used to reverse the guideline color if we near the edge of a lung FOV
                            while finish_final_chord != 1:    
                                #update pixel num
                                pixel_num = pixel_num + 1
                                #update chord measurement
                                pixel_chord_length = pixel_chord_length + 1
                                #continue to extend and make the fov guideline green
                                if save_fov == 1: #update FOV guideline if the user requests FOVs to be generated 
                                    r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                                    g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 255
                                    b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                                #check if we have finally reached a septa wall
                                if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == False and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 1] == False and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 2] == False and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 3] == False and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == True and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == True and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == True and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == True and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == True and \
                                    image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == True and first_cross == 1:
                                    finish_final_chord = 1
                                    if pixel_chord_length > 1:
                                        chord_image_name_list.append('img_'+str(count))
                                        chord_length_list.append(int(pixel_chord_length)*0.497885)#convert to micrometers and record
                                        total_wsi_chord_length = total_wsi_chord_length + (int(pixel_chord_length)*0.497885)
                                        total_wsi_chords_measured = total_wsi_chords_measured + 1
                                        
                                #if we reach a BV, BR or pleaural space, dont include chord, revert guideline color blue if save_fov==1
                                #check if guideline touches background
                                if fov_bg[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True or fov_bv[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True or fov_br[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                                    if save_fov == 1:                                        
                                        for revert_color_num in range (reverse_loc,pixel_num+1):
                                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_r_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_g_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_b_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                        #change color back to blue to indicate that a chord was not measured
                                        for blue_revert_num in range(last_crossing_loc,smp_size):
                                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 255  
                                    break
                                
                                #if we reach an edge of FOV, record chord instead of getting rid of measurement
                                if pixel_num - (smp_size/2) > fov_x/2 - 8:
                                    finish_final_chord = 1
                                    chord_image_name_list.append('img_'+str(count))
                                    chord_length_list.append(int(pixel_chord_length)*0.497885)#convert pixels to micrometers and record
                                    total_wsi_chord_length = total_wsi_chord_length + (int(pixel_chord_length)*0.497885)
                                    total_wsi_chords_measured = total_wsi_chords_measured + 1
                    
                    ##if end of guideline is inside septa wall and doesnt complete a crossing, and we have already completed a first cross
                    #then we must ensure guideline is blue on the right side of guideline
                    if first_cross == 1 and cross_1 == 1 and cross_2 == 0 and save_fov == 1:
                        for pixel_num_revert in range(last_cross_loc,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 255
                    
                    #################left side chord measurements
                    #we must also measure chords on left side of guideline
                    if first_cross == 1: #if we have at least one full crossing, we can measure the chord on the left side
                        finish_final_chord_left = 0 #used to stop the while loop once complete chords
                        first_cross_loc_left = first_cross_loc #location where first cross (left side (moving to the right)) occurs in FOV
                        pixel_num_left = first_cross_loc
                        pixel_chord_length= 0 #reset chord measurement
                        while finish_final_chord_left != 1:    
                            #update pixel num (moving right to left now)
                            pixel_num_left = pixel_num_left - 1
                            #update chord measurement
                            pixel_chord_length = pixel_chord_length + 1
                            #continue to extend and make the fov guideline green
                            if save_fov == 1: #update FOV guideline if the user requests FOVs to be generated 
                                r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] = 0
                                g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] = 255
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] = 0
                            #check if we have finally reached a septa wall
                            if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] == False and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left + 1] == False and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left + 2] == False and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left + 3] == False and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 1] == True and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 2] == True and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 3] == True and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 4] == True and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 5] == True and \
                                image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left - 6] == True:
                                
                                finish_final_chord_left = 1
                                if pixel_chord_length > 1:
                                    chord_image_name_list.append('img_'+str(count))
                                    chord_length_list.append(int(pixel_chord_length)*0.497885)#convert to micrometers and record
                                    total_wsi_chord_length = total_wsi_chord_length + (int(pixel_chord_length)*0.497885)
                                    total_wsi_chords_measured = total_wsi_chords_measured + 1
                                        
                            #if we reach a BV, BR or pleaural space, dont include chord, revert guideline color blue if save_fov==1
                            #check if guideline touches background
                            if fov_bg[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] == True or fov_br[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] == True or fov_bv[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_left] == True:
                                if save_fov == 1:                                        
                                    for revert_color_num in range (pixel_num_left, first_cross_loc_left+1):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_r_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_g_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + revert_color_num] = store_b_channel_accept[round(fov_x/2) - int(smp_size/2) + revert_color_num]
                                    #change color back to blue to indicate that a chord was not measured
                                    for blue_revert_num in range(0,first_cross_loc_left):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 255  
                                break
                                
                            #if we reach an edge of FOV, record chord instead of getting rid of measurement
                            if pixel_num_left + (smp_size/2) < - fov_x/2 + 8:
                                finish_final_chord = 1
                                chord_image_name_list.append('img_'+str(count))
                                chord_length_list.append(int(pixel_chord_length)*0.497885)#convert pixels to micrometers and record
                                total_wsi_chord_length = total_wsi_chord_length + (int(pixel_chord_length)*0.497885)
                                total_wsi_chords_measured = total_wsi_chords_measured + 1
                                #################left side chord measurements
                    
                    if first_cross == 0 and save_fov == 1: #we have never made a COMPLETE crossing, ensure guideline is fully blue
                        for pixel_num_revert in range(0,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 255 
                            
                    #if the reason we never made a crossing was because the guideline was fully inside an alveoli space, we can
                    #measure at least one chord 
                    outside_alveoli = 0
                    for alveoli_check in range(0,smp_size):
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + alveoli_check] == True:
                            outside_alveoli = 1
                            
                    #if we reach here and outside alveoli is still 0, then we are inside the alveoli and can record a chord measurement     
                    if outside_alveoli == 0:
                        chord_length_left = 0 #record left side of chord measurement
                        chord_length_right = 0 #record right side of chord measurement
                        chord_stop_left = 0 #stop point for left side
                        chord_stop_right = 0 #stop point for right side
                        do_not_record_chord = 0 #if == 1, we hit a BV, BR, BG and therefore will not record the chord measurement
                        
                        while chord_stop_right == 0 and do_not_record_chord == 0:
                            if save_fov == 1:
                                r_accepted_fov[round(fov_y/2), round(fov_x/2) + chord_length_right] = 0
                                g_accepted_fov[round(fov_y/2), round(fov_x/2) + chord_length_right] = 255
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) + chord_length_right] = 0
                            #stop once we reach a septa wall
                            if image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right - 1] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right - 2] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right - 3] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 1] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 2] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 3] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 4] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 5] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) + chord_length_right + 6] == True:
                                chord_stop_right = 1
                            chord_length_right = chord_length_right + 1
                            
                            #if we reach a BV, BR or pleaural space, dont include chord, revert guideline color blue if save_fov==1
                            #check if guideline touches background
                            if fov_bg[round(fov_y/2), round(fov_x/2) + chord_length_right] == True or \
                            fov_bv[round(fov_y/2), round(fov_x/2) + chord_length_right] == True or \
                            fov_br[round(fov_y/2), round(fov_x/2) + chord_length_right] == True:
                                if save_fov == 1:                                        
                                    for revert_color_num in range (0,chord_length_right+1):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) + revert_color_num] = store_r_channel_accept[round(fov_x/2) + revert_color_num]
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) + revert_color_num] = store_g_channel_accept[round(fov_x/2) + revert_color_num]
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) + revert_color_num] = store_b_channel_accept[round(fov_x/2) + revert_color_num]
                                    #change color back to blue to indicate that a chord was not measured
                                    for blue_revert_num in range(0,smp_size):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 255
                                do_not_record_chord = 1
                                
                            #if we reach an edge of the FOV we need to stop and hold the measurement to record, rather than delete it
                            if chord_length_right + 8 > fov_x/2:
                                chord_stop_right = 1
                        
                        #left side of chord measurement
                        while chord_stop_left == 0 and do_not_record_chord == 0:
                            if save_fov == 1:
                                r_accepted_fov[round(fov_y/2), round(fov_x/2) - chord_length_left] = 0
                                g_accepted_fov[round(fov_y/2), round(fov_x/2) - chord_length_left] = 255
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) - chord_length_left] = 0
                            #stop once we reach a septa wall
                            if image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left + 1] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left + 2] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left + 3] == False and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 1] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 2] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 3] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 4] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 5] == True and \
                            image_avlw[round(fov_y/2), round(fov_x/2) - chord_length_left - 6] == True:
                                chord_stop_left = 1
                            chord_length_left = chord_length_left + 1   
                    
                            #if we reach a BV, BR or pleaural space, dont include chord, revert guideline color blue if save_fov==1
                            #check if guideline touches background
                            if fov_bg[round(fov_y/2), round(fov_x/2) - chord_length_left] == True or \
                            fov_bv[round(fov_y/2), round(fov_x/2) - chord_length_left] == True or \
                            fov_br[round(fov_y/2), round(fov_x/2) - chord_length_left] == True:
                                if save_fov == 1:                                        
                                    for revert_color_num in range (0,chord_length_left+1):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) - revert_color_num] = store_r_channel_accept[round(fov_x/2) - revert_color_num]
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) - revert_color_num] = store_g_channel_accept[round(fov_x/2) - revert_color_num]
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) - revert_color_num] = store_b_channel_accept[round(fov_x/2) - revert_color_num]
                                    #change color back to blue to indicate that a chord was not measured
                                    for blue_revert_num in range(0,smp_size):
                                        r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 0
                                        b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + blue_revert_num] = 255
                                do_not_record_chord = 1
                                
                            #if we reach an edge of the FOV we need to stop and hold the measurement to record, rather than delete it
                            if chord_length_left + 8 > fov_x/2:
                                chord_stop_left = 1
                            
                        #if we reach here and do_not_record_chord is still 0, we can record a chord measurement
                        if do_not_record_chord == 0:
                            chord_image_name_list.append('img_'+str(count))
                            chord_length_list.append(int(chord_length_left + chord_length_right)*0.497885)#convert pixels to micrometers 
                            total_wsi_chord_length = total_wsi_chord_length + (int(chord_length_left + chord_length_right)*0.497885)
                            total_wsi_chords_measured = total_wsi_chords_measured + 1
                    
                    if save_fov == 1:
                        #now save our FOV with the guidline superimposed
                        rgb_fov_accepted = cv2.merge((r_accepted_fov,g_accepted_fov,b_accepted_fov))
                        tiff.imwrite(guideline_fov_save_location + '/' + 'img_' + str(count) + ".tif", rgb_fov_accepted)
                
                        #revert rgb_fov_accepted to original color 
                        for pixel_num_restore_2 in range (0,fov_x):
                                r_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_r_channel_accept[pixel_num_restore_2]
                                g_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_g_channel_accept[pixel_num_restore_2]
                                b_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_b_channel_accept[pixel_num_restore_2]
                        rgb_fov_accepted = cv2.merge((r_accepted_fov,g_accepted_fov,b_accepted_fov))
                            
                    crossing_total_count = crossing_total_count + crossings        
                    crossing_list.append(int(crossings)) #add number of crossings for the specific FOV to the list      
                else: #image was rejected
                    crossing_list.append('NaN')
                    
                    #superimpose a black guideline to indicate the FOV was rejected
                    if save_fov == 1:
                        #used these arrays to temporarily store the original colors where the guideline will be superimposed 
                        store_r_channel_reject = []
                        store_g_channel_reject = []
                        store_b_channel_reject = []
                        r_rejected_fov = r_patch_original
                        g_rejected_fov = g_patch_original
                        b_rejected_fov = b_patch_original
                        for pixel_num_4 in range (0,smp_size):
                            store_r_channel_reject.append(r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            store_g_channel_reject.append(g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            store_b_channel_reject.append(b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                            g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                            b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                        rgb_fov_rejected = cv2.merge((r_rejected_fov,g_rejected_fov,b_rejected_fov))
                        tiff.imwrite(guideline_fov_save_location + '/' + 'img_' + str(count) + ".tif", rgb_fov_rejected)
                        #now return FOV to its original color 
                        for pixel_num_restore_1 in range (0,smp_size):
                            r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_r_channel_reject[pixel_num_restore_1]
                            g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_g_channel_reject[pixel_num_restore_1]
                            b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_b_channel_reject[pixel_num_restore_1]
                        rgb_fov_rejected = cv2.merge((r_rejected_fov,g_rejected_fov,b_rejected_fov)) 
                
                #move to next FOV in x-direction and update count 
                count = count + 1 
                loc_x1 = int(loc_x1 + shift_x)
                loc_x2 = int(loc_x2 + shift_x)

        #move to next FOV in y-direction and reset loc_x
        loc_x1 = 1
        loc_x2 = fov_x
        loc_y1 = int(loc_y1 + shift_y)
        loc_y2 = int(loc_y2 + shift_y)

    df_info = pd.DataFrame({'FOV Name':FOV_name_list, 'FOV corner x1':pixel_loc_x1_list, 'FOV corner x2':pixel_loc_x2_list, 'FOV corner y1':pixel_loc_y1_list, 'FOV corner y2':pixel_loc_y2_list, 'ACC/REJ':acc_rej_list, '# Crossings': crossing_list})
    df_MLI = pd.DataFrame({'# ACC Images': acc_fov_count, '# Crossings': crossing_total_count, '# Chords':total_wsi_chords_measured, 'Average Chord Length':[(total_wsi_chord_length/total_wsi_chords_measured)],'MLI Score':[(acc_fov_count*smp_size)/(2.0084975*crossing_total_count)]})
    df_individual_chords = pd.DataFrame({'Image Chord': chord_image_name_list, 'Chord Measurements':chord_length_list})
    df_chord_data = pd.DataFrame({'# Chords Measured':total_wsi_chords_measured},index=[0])
    #df_average_chord = pd.DataFrame([(total_wsi_chord_length/total_wsi_chords_measured)], columns = ['Average Chord Length'])
    result = pd.concat([df, df_info, df_individual_chords, df_MLI], axis=1)
    result.to_csv(wsi_parent_path + '/' + wsi_name + '_MLI' + '/' + wsi_name+ "_Direct_MLI_Results.csv", index=False)
    
    # IN DIRC METHOD THE MLI SCORE IS Average Chord Length
    mliscore_value = df_MLI['Average Chord Length'].values[0]
    print("MLI Calculation is Completed")
    return mliscore_value

def indir_score(wsi_masks_folder_path, wsi_path, fov_x, fov_y, overlap_x, overlap_y, smp_size, save_fov):
    """
    Calculate the MLI (Mean Linear Intercept) score for Whole Slide Images (WSI).

    :param wsi_masks_folder_path: (str): Path to the folder where the WSI masks are located.
    :param fov_x: (int): FOV size in the x direction.
    :param fov_y: (int): FOV size in the y direction.
    :param overlap_x: (float): Overlap percentage in the x direction (between 0 and 0.99).
    :param overlap_y: (float): Overlap percentage in the y direction (between 0 and 0.99).
    :param smp_size: (int): Size of the guideline to superimpose.
    :param original_wsi_name: (str): Name of the original WSI.
    :param save_fov: (int): Flag indicating whether to save FOVs or not (0 - do not save, 1 - save).
    :return: None

    """

    wsi_mask_list = ['BG.tif', 'BR.tif', 'BV.tif', 'AVLW.tif']
    wsi_name, wsi_format = os.path.splitext(os.path.basename(wsi_path))
    wsi_parent_path = os.path.dirname(wsi_path)


    # load the respective WSI masks
    BG_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[0]), 0)

    # Ensure the user input values are not invalid
    if fov_x > BG_BW_WSI.shape[0]:
        raise ValueError('Error: fov_x cannot be greater than the image x length.')

    if fov_y > BG_BW_WSI.shape[1]:
        raise ValueError('Error: fov_y cannot be greater than the image y length.')

    if overlap_x > 0.99 or overlap_y > 0.99 or overlap_x < 0 or overlap_y < 0:
        raise ValueError('Error: overlap cannot be greater than 0.99 (99%) or less than 0 (no overlap).')

    if save_fov < 0 or save_fov > 1:
        raise ValueError('Error: save_fov must be either 0 (do not save FOVs) or 1 (save FOVs).')

    BR_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[1]), 0)
    BV_BW_WSI = cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[2]), 0)
    AVLW_BW_WSI=cv2.imread(os.path.join(wsi_masks_folder_path, wsi_mask_list[3]), 0)

    print("Starting MLI Calculation using Indirect Method")
    mli_folder = os.path.join(wsi_parent_path, f"{wsi_name}_MLI")        
    os.makedirs(mli_folder, exist_ok=True)

    if save_fov == 1:
        # load original WSI and split into rgb channels
        ORIGINAL_WSI = io.imread(wsi_path)
        ORIGINAL_WSI = ORIGINAL_WSI[:, :, :3]  # remove Alpha channel
        # split into RGB color channels
        r_ORIGINAL_WSI, g_ORIGINAL_WSI, b_ORIGINAL_WSI = cv2.split(ORIGINAL_WSI)

        # create our file to store FOVs with guidelines superimposed
        guideline_fov_save_location = wsi_parent_path + '/' + wsi_name + '_MLI' + '/' + wsi_name + '_FOV_Guideline_Indirect_MLI'
        os.makedirs(guideline_fov_save_location, exist_ok=True)

    #count used for FOV image numbering (ex. img_1, img_2...etc)
    count = 1 
    #loc_x1 and loc_x2 used for tracking horizontal pixel location of FOV with respect to WSI 
    loc_x1 = 1
    loc_x2 = fov_x
    #loc_y1 and loc_y2 used for tracking vertical pixel location of FOV with respect to WSI 
    loc_y1 = 1
    loc_y2 = fov_y
    #shift_x and shift_y determines change in horizontal/vertical pixel location for each FOV, respectively
    #ex. if overlap_x is 30% overlap, then the next FOV image in x direction must be shifted 70%. shift_x = # pixels to shift
    shift_x = fov_x*(1-overlap_x)
    shift_y = fov_y*(1-overlap_y)

    #create dataframe and lists to track information
    original_naming = [wsi_name]
    df = pd.DataFrame(original_naming, columns = ['Original Image Name'])
    FOV_name_list = []#track FOV image number
    pixel_loc_x1_list = []#track x1 pixel location
    pixel_loc_x2_list = []#track x2 pixel location
    pixel_loc_y1_list = []#track y1 pixel location
    pixel_loc_y2_list = []#track y2 pixel location
    acc_rej_list = [] #initialize a list to store whether the FOV was rejected or accepted
    #if the image is ACC:
    crossing_list = [] #create a list to track the # crossings per FOV
    crossing_total_count = 0 #keep track of overal # crossings for entire FOV set
    acc_fov_count = 0 #keep track of number of accepted FOVs
    
    #extract the FOVs based on FOV size and overlap
    for i in tqdm.tqdm(range(0, BG_BW_WSI.shape[0], int(fov_y * (1 - overlap_y))), desc="Processing FOVs"):
        for j in range(0, BG_BW_WSI.shape[1], int(fov_x*(1-overlap_x))):  #Steps of fov_x with overlap_x
            #grab FOV for each of the 4 masks
            single_patch_BG = BG_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_BR = BR_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_BV = BV_BW_WSI[i:i+fov_y, j:j+fov_x]
            single_patch_AVLW = AVLW_BW_WSI[i:i+fov_y, j:j+fov_x]
            
            if save_fov == 1:
                #grab each color channel FOV from the original wsi 
                r_patch_original = r_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]
                g_patch_original = g_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]
                b_patch_original = b_ORIGINAL_WSI[i:i+fov_y, j:j+fov_x]

            # only keep FOVs that are same shape as fov_x,fov_y, remove smaller patches around the sides
            if single_patch_BG.shape[0] == fov_y and single_patch_BG.shape[1] == fov_x:
                #track image number, x and y pixel locations (dataframe)
                FOV_name_list.append('img_'+str(count))
                pixel_loc_x1_list.append(int(loc_x1))
                pixel_loc_x2_list.append(int(loc_x2))
                pixel_loc_y1_list.append(int(loc_y1))
                pixel_loc_y2_list.append(int(loc_y2))            

                #now superimpose guidline onto FOV masks (BG, BR, BV, AVLW) to either accept or reject the specific FOV location            
                #first convert FOVs into datatype bool
                fov_bg = np.array(single_patch_BG, dtype=bool)
                fov_br = np.array(single_patch_BR, dtype=bool)
                fov_bv = np.array(single_patch_BV, dtype=bool)
                #look at every pixel that composes the guidline and determine if there is a "True" value, and reject if "True":
                rej_trigger = 0 #if the item is rejected, no need to determine # of crossings
                #check if guidline touches the background
                for pixel_num in range (0,smp_size):
                    if fov_bg[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                        rej_trigger = 1
                        acc_rej_list.append('REJ_BG')
                        break
                #check if guideline touches a bronchi
                if rej_trigger == 0: #if the FOV has not been rejected because of background
                    for pixel_num in range (0,smp_size):
                        if fov_br[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                            rej_trigger = 1
                            acc_rej_list.append('REJ_BR')
                            break
                #check if guidline touches a blood vessel
                if rej_trigger == 0: #if the FOV has not been rejected because of background and BR
                    for pixel_num in range (0,smp_size):
                        if fov_bv[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True:
                            rej_trigger = 1
                            acc_rej_list.append('REJ_BV')
                            break

                #if we reach this point and rej_trigger is still 0, the image is accepted and we can calculate # crossings
                if rej_trigger == 0:
                    acc_rej_list.append('ACC')
                    acc_fov_count = acc_fov_count + 1 #update our count for the number of accepted FOVs
                    cross_1 = 0 #keep track of transition from False to True (avleolar/ductile space into septa wall)
                    cross_2 = 0 #keep track of transition from True to False (septa wall into avleolar/ductile space)
                    crossings = 0 #keep track of number of crossings that occured in the FOV
                    first_cross = 0 #track when the first crossing occurs in the image
                    first_cross_loc = 0 #record location where first crossing occurs in the FOV
                    last_cross_loc = 0 #track where the last crossing (left side) occured
                    
                    if save_fov == 1:
                        #used these arrays to temporarily store the original colors where the guideline will be superimposed 
                        store_r_channel_accept = []
                        store_g_channel_accept = []
                        store_b_channel_accept = []
                        r_accepted_fov = r_patch_original
                        g_accepted_fov = g_patch_original
                        b_accepted_fov = b_patch_original

                        #superimpose a blue guideline and then change to red for the areas that have a crossing later
                        #include the entire width to store and restore
                        for pixel_num_5 in range(0,fov_x):
                            store_r_channel_accept.append(r_accepted_fov[round(fov_y/2), pixel_num_5])
                            store_g_channel_accept.append(g_accepted_fov[round(fov_y/2), pixel_num_5])
                            store_b_channel_accept.append(b_accepted_fov[round(fov_y/2), pixel_num_5])
                        for pixel_num_6 in range(0,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_6] = 255
                    
                    #perform post processing on the AVLW image to remove noise:
                    #convert to datatype bool
                    fov_avlw = np.array(single_patch_AVLW, dtype=bool)
                    #remove small white objects from alveoli space
                    image_avlw_bwareopen = morphology.remove_small_objects(fov_avlw, 250)#pixel limit trial/error
                    #NOT (~) binary image
                    image_avlw_not = ~image_avlw_bwareopen
                    #remove small black objects from septa walls
                    image_avlw_not_bwareaopen = morphology.remove_small_objects(image_avlw_not, 500)#pixel limit trial/error
                    #convert back to original binary image with black/white small objects removed
                    image_avlw = ~image_avlw_not_bwareaopen 

                    #iterate through the superimposed guidline to determine crossings
                    for pixel_num in range (0,smp_size):
                        #if transition from pixel_num to pixel_num + 1 is from alveolar/ductile space to septa wall,
                        #and this is the first crossing found, update cross_1:
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == True and first_cross == 0:
                            cross_1 = 1
                            first_cross_loc = pixel_num #store location where first crossing begins
                            
                        #if transition from pixel_num to pixel_num + 1 is from alveolar/ductile space to septa wall,
                        #and this is for further crossings, we can now start to RECORD chord lengths
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num - 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == True and first_cross == 1:
                            cross_1 = 1
                            
                        ##if cross_1=1 and cross_2=0, we can now make this area of the guideline red
                        if cross_1 == 1 and cross_2 == 0 and save_fov == 1:
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 255
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                            #no need to change green channel, it has already been set to 0 before 

                        #if transition from pixel_num to pixel_num + 1 is from septa wall into alveolar/ductile space
                        #and we already crossed into septa region from avleolar/ductile space previously, update cross_2:
                        if image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] == True and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 1] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 2] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 3] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 4] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 5] == False and \
                        image_avlw[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num + 6] == False and cross_1 == 1:
                            cross_2 = 1
                            first_cross = 1                          
                        
                        #if we have made a full transition through a septa wall (cross_1 = 1 and cross_2 = 1), crossings++
                        if cross_1 == 1 and cross_2 == 1: #fully crossed a septa wall
                            crossings = crossings + 1
                            cross_1 = 0 #reset to find next crossing
                            cross_2 = 0 #reset to find next crossing
                            last_crossing_loc = pixel_num #location of right-side of last crossing 
                        
                        #if we have made a full crossing then we can make the chord blue
                        if cross_1 == 0 and cross_2 == 0 and first_cross == 1:
                            if save_fov == 1: #update FOV guideline if the user requests FOVs to be generated                           
                                g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 0
                                b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num] = 255
                    
                    #if we are in a crossing but reach edge of guideline, revert that portion to blue
                    if first_cross == 1 and cross_1 == 1 and cross_2 == 0 and save_fov == 1:
                        for pixel_num_revert in range(last_crossing_loc,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 255 
                    
                    if first_cross == 0 and save_fov == 1: #we have never made a COMPLETE crossing, ensure guideline is fully blue
                        for pixel_num_revert in range(0,smp_size):
                            r_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            g_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 0
                            b_accepted_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_revert] = 255 
                                
                    if save_fov == 1:
                        #now save our FOV with the guidline superimposed
                        rgb_fov_accepted = cv2.merge((r_accepted_fov,g_accepted_fov,b_accepted_fov))
                        tiff.imwrite(guideline_fov_save_location + '/' + 'img_' + str(count) + ".tif", rgb_fov_accepted)
                
                        #revert rgb_fov_accepted to original color 
                        for pixel_num_restore_2 in range (0,fov_x):
                                r_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_r_channel_accept[pixel_num_restore_2]
                                g_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_g_channel_accept[pixel_num_restore_2]
                                b_accepted_fov[round(fov_y/2), pixel_num_restore_2] = store_b_channel_accept[pixel_num_restore_2]
                        rgb_fov_accepted = cv2.merge((r_accepted_fov,g_accepted_fov,b_accepted_fov))
                            
                    crossing_total_count = crossing_total_count + crossings        
                    crossing_list.append(int(crossings)) #add number of crossings for the specific FOV to the list      
                else: #image was rejected
                    crossing_list.append('NaN')
                    
                    #superimpose a black guideline to indicate the FOV was rejected
                    if save_fov == 1:
                        #used these arrays to temporarily store the original colors where the guideline will be superimposed 
                        store_r_channel_reject = []
                        store_g_channel_reject = []
                        store_b_channel_reject = []
                        r_rejected_fov = r_patch_original
                        g_rejected_fov = g_patch_original
                        b_rejected_fov = b_patch_original
                        for pixel_num_4 in range (0,smp_size):
                            store_r_channel_reject.append(r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            store_g_channel_reject.append(g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            store_b_channel_reject.append(b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4])
                            r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                            g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                            b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_4] = 0
                        rgb_fov_rejected = cv2.merge((r_rejected_fov,g_rejected_fov,b_rejected_fov))
                        tiff.imwrite(guideline_fov_save_location + '/' + 'img_' + str(count) + ".tif", rgb_fov_rejected)
                        #now return FOV to its original color 
                        for pixel_num_restore_1 in range (0,smp_size):
                            r_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_r_channel_reject[pixel_num_restore_1]
                            g_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_g_channel_reject[pixel_num_restore_1]
                            b_rejected_fov[round(fov_y/2), round(fov_x/2) - int(smp_size/2) + pixel_num_restore_1] = store_b_channel_reject[pixel_num_restore_1]
                        rgb_fov_rejected = cv2.merge((r_rejected_fov,g_rejected_fov,b_rejected_fov)) 
                
                #move to next FOV in x-direction and update count 
                count = count + 1 
                loc_x1 = int(loc_x1 + shift_x)
                loc_x2 = int(loc_x2 + shift_x)

        #move to next FOV in y-direction and reset loc_x
        loc_x1 = 1
        loc_x2 = fov_x
        loc_y1 = int(loc_y1 + shift_y)
        loc_y2 = int(loc_y2 + shift_y)

    df_info = pd.DataFrame({'FOV Name':FOV_name_list, 'FOV corner x1':pixel_loc_x1_list, 'FOV corner x2':pixel_loc_x2_list, 'FOV corner y1':pixel_loc_y1_list, 'FOV corner y2':pixel_loc_y2_list, 'ACC/REJ':acc_rej_list, '# Crossings': crossing_list})
    df_MLI = pd.DataFrame({'# ACC Images': acc_fov_count, 'Total Crossings': crossing_total_count,'MLI Score':[(acc_fov_count*smp_size)/(crossing_total_count*2.00849789)]})
    result = pd.concat([df, df_info,df_MLI], axis=1)
    result.to_csv(wsi_parent_path + '/' + wsi_name + '_MLI' + '/' + wsi_name + "_Indirect_MLI_Results.csv", index=False)
    mliscore_value = df_MLI['MLI Score'].values[0]
    print("MLI Calculation is Completed")
    return mliscore_value
