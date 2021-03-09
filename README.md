# FeaturePoints
Draws correspondence between feature points in two given image.

## Arguments:
* n - number of the best feature points
* m - best corresponding feature points for a selected feature point (will be clear next).

## How it works:
1. Compute the best n SIFT points on the two images. The best is computed according to the 
SIFT descriptor 
2. The user can select a feature point, p, from the image L and the program will show the m
feature points on the image R, which are the most similar points to p according to the SIFT 
descriptor 
3. It then computes the maximum matching cover among the feature points of the two images, 
write it to a text file, and display it to the user.

For more information check the results doc.