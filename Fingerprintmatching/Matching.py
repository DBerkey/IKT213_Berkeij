"""
Author: Douwe Berkeij
Date: 22-09-2025
"""

import cv2

def orb_feature_BFMatcher(img1, img2):
    orb = cv2.ORB_create()
 
    # Detect and compute keypoints and descriptors for two images
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Find good matches using the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 20:
        matched = True
    else:
        matched = False

    # Draw the top matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, matched

def SIFT_feature_BFMatcher(img1, img2):
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for two images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # find good matches using the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 20:
        matched = True
    else:
        matched = False

    # Draw the top matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, matched

def SIFT_feature_FLANNMatcher(img1, img2):
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for two images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters (KD-tree for SIFT)
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)  # Number of checks for nearest neighbors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
 
    # KNN Match
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
 
    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) >= 20:
        matched = True
    else:
        matched = False

    # Draw the top matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, matched

if __name__ == "__main__":
    fingerprint_images_folder = "Fingerprintmatching/data_check"
    img1 = cv2.imread(f"{fingerprint_images_folder}/same_1/101_6.tif", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"{fingerprint_images_folder}/same_1/101_7.tif", cv2.IMREAD_GRAYSCALE)
    matched_image_BFMatcher, matched_BFMatcher = orb_feature_BFMatcher(img1, img2)
    matched_image_SIFT, matched_SIFT = SIFT_feature_BFMatcher(img1, img2)

    print(f"ORB BFMatcher matched same: {matched_BFMatcher}")
    print(f"SIFT BFMatcher matched same: {matched_SIFT}")

    # cv2.imshow("ORB Feature Matching with BFMatcher", matched_image_BFMatcher)
    # cv2.imshow("SIFT Feature Matching with BFMatcher", matched_image_SIFT)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # both methods work for matching the fingerprints, I dont see a significant difference

    img3 = cv2.imread(f"{fingerprint_images_folder}/different_4/106_6.tif", cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(f"{fingerprint_images_folder}/different_4/106_7.tif", cv2.IMREAD_GRAYSCALE)
    matched_image_BFMatcher_diff, matched_BFMatcher_diff = orb_feature_BFMatcher(img3, img4)
    matched_image_SIFT_diff, matched_SIFT_diff = SIFT_feature_BFMatcher(img3, img4)

    print(f"ORB BFMatcher matched different: {matched_BFMatcher_diff}")
    print(f"SIFT BFMatcher matched different: {matched_SIFT_diff}")

    # cv2.imshow("ORB Feature Matching with BFMatcher (Different)", matched_image_BFMatcher_diff)
    # cv2.imshow("SIFT Feature Matching with BFMatcher (Different)", matched_image_SIFT_diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Only the ORB method found that these two fingerprints are different, SIFT said they were the same

    img5 = cv2.imread(f"Fingerprintmatching/UiA front1.png", cv2.IMREAD_GRAYSCALE)
    img6 = cv2.imread(f"Fingerprintmatching/UiA front3.jpg", cv2.IMREAD_GRAYSCALE)
    matched_image_BFMatcher_diff, matched_BFMatcher_diff = orb_feature_BFMatcher(img5, img6)
    matched_image_SIFT_diff, matched_SIFT_diff = SIFT_feature_BFMatcher(img5, img6)

    print(f"ORB BFMatcher matched UiA: {matched_BFMatcher_diff}")
    print(f"SIFT BFMatcher matched UiA: {matched_SIFT_diff}")

    # cv2.imshow("ORB Feature Matching with BFMatcher (Different)", matched_image_BFMatcher_diff)
    # cv2.imshow("SIFT Feature Matching with BFMatcher (Different)", matched_image_SIFT_diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Only SIFT found that these two images are the same, ORB said they were different

    matched_image_SIFT_FLANN, matched_SIFT_FLANN = SIFT_feature_FLANNMatcher(img1, img2)
    print(f"SIFT FLANNMatcher matched same: {matched_SIFT_FLANN}")

    # cv2.imshow("SIFT Feature Matching with FLANNMatcher", matched_image_SIFT_FLANN)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # FLANN also works well, similar results to BFMatcher with SIFT for these images

    matched_image_SIFT_FLANN_diff, matched_SIFT_FLANN_diff = SIFT_feature_FLANNMatcher(img3, img4)
    print(f"SIFT FLANNMatcher matched different: {matched_SIFT_FLANN_diff}")

    # cv2.imshow("SIFT Feature Matching with FLANNMatcher (Different)", matched_image_SIFT_FLANN_diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # instead of what BFMatcher did, FLANN found these two different fingerprints correctly as different

    matched_image_SIFT_FLANN_UiA, matched_SIFT_FLANN_UiA = SIFT_feature_FLANNMatcher(img5, img6)
    print(f"SIFT FLANNMatcher matched UiA: {matched_SIFT_FLANN_UiA}")

    cv2.imshow("SIFT Feature Matching with FLANNMatcher (UiA)", matched_image_SIFT_FLANN_UiA)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # FLANN also found these two images as the same, similar to BFMatcher with SIFT
