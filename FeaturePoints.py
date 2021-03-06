import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def find_closest_key_point(key_points, x, y):
    min_distance = 99999999
    return_key = key_points[0]
    for key in key_points_1:
        if abs(key.pt[1] - y) + abs(key.pt[0] - x) < min_distance:
            min_distance = abs(key.pt[1] - y) + abs(key.pt[0] - x)
            return_key = key
    return return_key


# this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    if event == 2:
        print([x,y])
        m, img1, img2, key_points_1, key_points_2, descriptors_1, descriptors_2 = params
        needed_KeyPoint = np.array([find_closest_key_point(key_points_1, x, y)])
        bf2 = cv2.BFMatcher(cv2.NORM_L1, False)
        matches2 = bf2.knnMatch(descriptors_1,descriptors_2, m)
        new_matches = []
        for mat in matches2:
            if(needed_KeyPoint[0] == key_points_1[mat[0].queryIdx]):
                new_matches = [mat]
                break
        print(len(new_matches))
        img4 = cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, new_matches, img2, flags=2)
        plt.title("FeaturePoints matches")
        plt.imshow(img4), plt.show()


if __name__ == "__main__":
    n = int(sys.argv[1])  # number of the best features points
    m = int(sys.argv[2])  # best corresponding feature points for a selected feature point
    img1 = cv2.imread('img1.png')
    img2 = cv2.imread('img2.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(n) # where n is the limit

    key_points_1, descriptors_1 = sift.detectAndCompute(img1, None)
    key_points_2, descriptors_2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    coordinates = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        coordinates.append((key_points_1[img1_idx].pt, key_points_2[img2_idx].pt))

    with open('MatchCoordinatesResults.txt', 'w') as fp:
        fp.write('IMG-1\t\t\t\t\t\t\t\t\t\t\t\tIMG-2\n')
        fp.write('\n'.join('%s %s' % x for x in coordinates))

    img3 = cv2.drawMatches(img1, key_points_1, img2, key_points_2, matches, img2, flags=2)
    plt.title("MAXIMUM MATCHING COVER")
    plt.imshow(img3), plt.show()

    img1_with_blobs = cv2.drawKeypoints(img1, key_points_1, np.array([]),
                                        (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.namedWindow('please choose feature point', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("please choose feature point", mouse_callback, (m, img1, img2, key_points_1, key_points_2,
                                                                         descriptors_1, descriptors_2))
    cv2.imshow("please choose feature point", img1_with_blobs)
    cv2.waitKey()