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
    Y_MAX  = 235;
    Cr_MIN = 133;
    Cr_MAX = 180;
    Cb_MIN = 75;
    Cb_MAX = 135;
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
    Mat cameraFeed,skinMat,fore,back;
    skindetector mySkinDetector;
    vector<vector<Point> > contours,bContours;
	vector<Vec4i> hierarchy,bHierarchy;
	cv:RNG rng;	
	KalmanFilter KF(4, 2, 0);
	bool flag = false;
	int numberOfPeople = 0;
	cv::Rect window;
	int windowHeight = 480, windowWidth = 320;


	// background detection properties
	cv::BackgroundSubtractorMOG2 bg;
    bg.nmixtures = 5;
    bg.bShadowDetection = false;
	bg.history = 500;
	bg.varThreshold = 16;
	
	//Init Kalman filter configurations
	initFunctions(KF);

	//open capture object at location zero (default location for webcam)
	//set height and width of capture frame


	capture.open(0);
	

	capture.set(CV_CAP_PROP_FRAME_WIDTH,windowWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,windowHeight);	
	//capture.open("d:\\test.avi");
	//resizeWindow("Original Image",320,480);
	window.x = 0;
	window.y = 0;
	window.width = windowWidth;
	window.height = windowHeight;

    //Create a structuring element
    int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    

	Scalar color = Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) );

    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop

	while(1){
		double contourArea = 0;

        //store image to matrix
        capture.read(cameraFeed);

        //show the current image
        skinMat = mySkinDetector.getSkin(cameraFeed);
		bg.operator ()(cameraFeed,fore);

		cv::erode(skinMat, skinMat, element);
		cv::dilate(skinMat, skinMat, element);
		

			
		cv::erode(fore,fore,element);
		cv::dilate(fore,fore,element);
		cv::findContours(fore,bContours,bHierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
		
		//Get Skin contours
		cv::findContours(skinMat,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
        	
		for(int i = 0;i < bContours.size();i++){
			if(cv::contourArea(bContours[i]) > 500){
				cv::drawContours( cameraFeed, bContours, i,cv::Scalar(0,0,255), 1.5, 1, bHierarchy, CV_16SC1, cv::Point() );
			}
		}

		
	    int j = -1;
		for( int i = 0; i< contours.size(); i++) {
			if(cv::contourArea(contours[i]) > 700){
				if(cv::contourArea(contours[i]) > contourArea){
					contourArea = cv::contourArea(contours[i]);
					j = i;
				}
				drawContours( skinMat, contours, i,color, 1.5, 1, hierarchy, CV_16SC1, Point() );
			}
		}

		
		Moments mu;	
		Point2f mc = -1;

		if(j >= 0){
			mu = moments( contours[j], false );
			mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 ); 
			drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
		}
 
		if(mc.inside(window) && flag == false){
			numberOfPeople++;
			flag = true;
			printf("%d ",numberOfPeople);
		}

		if(!mc.inside(window))
			flag = false;

		
		imshow("Skin Image",skinMat);
	    imshow("Original Image",cameraFeed);
        waitKey(16);
    }
    return 0;
}

