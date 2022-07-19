#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

cv::Mat img, imgGray, img2, imgGray2;
vector<cv::KeyPoint> keypoints1, keypoints2; 
Mat descriptors1, descriptors2;
vector<DMatch> matches;

// create a function that takes vector of keypoints and image then draws them on the image
void drawKeypoints(vector<KeyPoint> keypoints, Mat &img, Scalar color)
{
    for (int i = 0; i < keypoints.size(); i++)
    {
        circle(img, keypoints[i].pt, 2, color, -1);
    }
}
// create a function that takes vector of keypoints and image then detects and finds the descriptors for them
void detectDescribe(vector<KeyPoint> keypoints, Mat &img, Mat &descriptors)
{
    // detect keypoints
    Ptr<Feature2D> detector = ORB::create();
    detector->detect(img, keypoints);
    // compute descriptors
    Ptr<Feature2D> descriptor = ORB::create();
    descriptor->compute(img, keypoints, descriptors);

    // draw keypoints on image
    drawKeypoints(keypoints, img, Scalar(0, 255, 0));
    cv::imshow("image", img);
    cv::waitKey(0);
}
// create a function to match the descriptors of the two images
void matchDescriptors(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches)
{
    // create a matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    // match descriptors
    matcher->match(descriptors1, descriptors2, matches);
    Mat imgMatches;
    drawMatches(img, keypoints1, img2, keypoints2, matches, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::waitKey(0);
}
 
int main(){

    vector<Mat> images;

    for (int i = 1; i < 11; i++)
    {
        stringstream ss;
        ss << "/home/bhavik/projects/sensorFusion/feature_tracking/data/0" << i << ".jpg";
        string filename = ss.str();
        Mat img = cv::imread(filename);
        images.push_back(img);
    }

    vector<vector<KeyPoint>> keypoints_buffer;

    for (int i = 0; i < 10; i++)
    {
        vector<KeyPoint> keypoints;
        Ptr<Feature2D> detector = ORB::create();
        detector->detect(images[i], keypoints);
        keypoints_buffer.push_back(keypoints);
    }

    vector<Mat> descriptors_buffer;

    for (int i = 0; i < 10; i++)
    {
        Mat descriptors;
        Ptr<Feature2D> descriptor = ORB::create();
        descriptor->compute(images[i], keypoints_buffer[i], descriptors);
        descriptors_buffer.push_back(descriptors);
    }

    for (int i = 0; i < 10; i++)
    {
        Mat img = images[i];
        drawKeypoints(keypoints_buffer[i], img, Scalar(255, 255, 0));
        cv::imshow("image", img);
        cv::waitKey(0);
    }
    
    return 0;

}