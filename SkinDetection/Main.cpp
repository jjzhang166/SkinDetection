#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

#define drawCross( img, center, color, d )\
line(img, Point(center.x - d, center.y - d), Point(center.x + d, center.y + d), color, 2, CV_AA, 0);\
line(img, Point(center.x + d, center.y - d), Point(center.x - d, center.y + d), color, 2, CV_AA, 0 )\


//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = {0,0};
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0,0,0,0);


//  skindetector.cpp
//  oo
////
////  Created by João Lucas Sisanoski on 03/10/14.
////  Copyright (c) 2014 João Lucas Sisanoski. All rights reserved.
////
//
using namespace std;
class skindetector
{
public:
    skindetector(void);
    ~skindetector(void);
    
    cv::Mat getSkin(cv::Mat input);
    
private:
    int Y_MIN;
    int Y_MAX;
    int Cr_MIN;   
	int Cr_MAX;
    int Cb_MIN;
    int Cb_MAX;
};skindetector::skindetector(void)
{
    //YCrCb threshold
    // You can change the values and see what happens
    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 160;
    Cb_MIN = 70;
    Cb_MAX = 127;
}

skindetector::~skindetector(void)
{
}

////this function will return a skin masked image
cv::Mat skindetector::getSkin(cv::Mat input)
{
    cv::Mat skin;
    //first convert our RGB image to YCrCb
    
    cv::cvtColor(input,skin,cv::COLOR_BGR2YCrCb);
    
    //uncomment the following line to see the image in YCrCb Color Space
    //cv::imshow("YCrCb Color Space",skin);
    
    //filter the image in YCrCb color space
    cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
    return skin;
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed){
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	int xMax = 0,yMax = 0,xMin = 50000,yMin = 50000;
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours
	
	//if contours vector is not empty, we have found some objects
	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	if(objectDetected){
		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		vector< vector<Point> > largestContourVec;
		largestContourVec.push_back(contours[contours.size()-1]);
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.

		//this code get the max and minimal points at the contours

		for(int i = 0;i < largestContourVec[0].size();i++){		
			if(xMax < largestContourVec[0][i].x){
				xMax = largestContourVec[0][i].x;
			}

			if(xMin > largestContourVec[0][i].x){
				xMin = largestContourVec[0][i].x;
			}

			if(yMax < largestContourVec[0][i].y){
				yMax = largestContourVec[0][i].y;
			}

			if(yMin > largestContourVec[0][i].y){
				yMin = largestContourVec[0][i].y;
			}
		}

		//printf("%d\n",largestContourVec[0].size());
		//printf("Xmax: %d\nXmin: %d\nYMax: %d\nYMin: %d\n",xMax,xMin,yMax,yMin);

		//update the objects positions by changing the 'theObject' array values
	}
	//make some temp x and y variables so we dont have to type out so much
	//int x = theObject[0];
	//int y = theObject[1];
	
	//draw some crosshairs around the object
	
	//circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
	
	line(cameraFeed,Point(xMin,yMax),Point(xMax,yMax),Scalar(0,0,255),1);
	line(cameraFeed,Point(xMax,yMax),Point(xMax,yMin),Scalar(0,0,255),1);
	line(cameraFeed,Point(xMax,yMin),Point(xMin,yMin),Scalar(0,0,255),1);
	line(cameraFeed,Point(xMin,yMin),Point(xMin,yMax),Scalar(0,0,255),1);
	
	//write the position of the object to the screen
}

void initFunctions(KalmanFilter KF){
	KF.statePre.at<float>(0) = 0;
    KF.statePre.at<float>(1) = 0;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;

    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
    KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
}

int main(int argc, const char *argv[]) {
    VideoCapture capture;
    Mat cameraFeed;
    skindetector mySkinDetector;
    Mat skinMat;
    vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double contourArea;
	cv:RNG rng;	
	KalmanFilter KF(4, 2, 0);
	Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
	initFunctions(KF);
 



	//open capture object at location zero (default location for webcam)

    capture.open(0);

    //set height and width of capture frame
    capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);



     //Create a structuring element
    int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    

	Scalar color = Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) );

    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop
    while(1){

        //store image to matrix
        capture.read(cameraFeed);

        //show the current image
        skinMat= mySkinDetector.getSkin(cameraFeed);
        cv::erode(skinMat, skinMat, element);
		
		cv::findContours(skinMat,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
      
		vector<Moments> mu(contours.size() );

		for( int i = 0; i< contours.size(); i++) {
			if(cv::contourArea(contours[i]) > 500){
				drawContours( skinMat, contours, i,color, 1, 4, hierarchy, 0, Point() );
				mu[i] = moments( contours[i], false );
			}
		}

	  vector<Point2f> mc( contours.size() );
      for( size_t i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
	  
	  for(size_t i = 0; i < mc.size(); i++)
        {
		drawCross(cameraFeed, mc[i], Scalar(255, 0, 0), 5);
          measurement(0) = mc[i].x;
          measurement(1) = mc[i].y;
        }

	  Mat estimated = KF.correct(measurement);
      Point statePt(estimated.at<float>(0),estimated.at<float>(1));

	  printf("x: %d\ny: %d\n",estimated.at<float>(0),estimated.at<float>(1));
	  drawCross(cameraFeed, statePt, Scalar(128, 128, 128), 5);
	  

		//searchForMovement(skinMat,cameraFeed);

		imshow("Skin Image",skinMat);
	    imshow("Original Image",cameraFeed);
        waitKey(30);
    }
    return 0;
}
