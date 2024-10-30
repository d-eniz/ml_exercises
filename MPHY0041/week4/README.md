## Exercises Week 4

1)	For this exercise it is encouraged that you will install the python package scikit-image (skimage). This package contains advanced image processing and manipulation functions – again, we will not use many of those, besides loading the images, but knowing your way around what skimage has to offer can be useful (filters, feature extraction etc.).

2)	Using the imread() function in skimage load the images [./Tumor/TCGA_CS_6186_20000601_19.tif](./week4/Tumor/TCGA_CS_6186_20000601_19.tif) and [‘./Mask/ TCGA_CS_6186_20000601_19_mask.tif’](./week4/Mask/ TCGA_CS_6186_20000601_19_mask.tif) which are the single slice of an brain image (256x256 pixels in RGB [i.e., 3 channels]) and the expert annotated mask (256x256 pixel), respectively. Plot both images within python using (e.g., matplotlib’s imshow()) function.

3)	From this image, we will now create a training set for our classifier. This is done by extracting 5x5 patches from the image. Sklearn has a convenient function for this (extract_patches_2d). Use it to extract 1000 5x5 patches. (Hint: Make sure that the same patches are extracted for the brain image and the mask, e.g., by adding the mask to the image and creating a 256x256x4 array). Flatten the patches into vectors of size 75 = (5x5x3). After this step you should have one matrix (X) of size (1000x75) and one vector of size 1000 (Y). Y should simply contain a class label (0 or 1) for each patch.

4)	Train a Random Forest Classifier with 100 trees and min_samples_split=5 on this dataset.

5)	Use the same input image to create your test set. This time we want all 5x5 patches. (Hint: you can add padding to your image array using numpy’s pad function. This ensures that our result at the end will have the same dimension as our input. E.g, for 5x5 patches we want to add padding of 2 rows and columns to each channel. Otherwise, we will be losing a few pixels). Use the classifier you trained in (4) to make predictions for each patch. Turn the predicted probabilities for class 1 (‘tumour’) into a 256x256 vector and plot the result.

6)	Of course, we would like to have a classifier that works on out-of-sample images. To do this, use all the images in the ‘./Tumor/’ folder (and their corresponding masks). From each of them extract 2000 patches of size 5x5. Use the resulting training data to train a Random Forest Classifier and predict the data for the five examples in the ‘./Test_Tumor/’ folder. Plot your result and compute the DICE coefficient between the actual mask and your predicted result (e.g., convert the probabilities to class labels using a probability cutoff at 0.5 – or use the predict instead of the predict_proba function of your classifier).

7)	Challenge: Play around with size of the forest (warning: training can get very slow with lots of trees), dimension of the patches, and number of patches per training image. Do you see and improvements?
