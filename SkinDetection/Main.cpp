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
    Y_MIN  = 30;
    Y_MAX  = 235;
    Cr_MIN = 133;
    Cr_MAX = 170;
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

Mat getImagePart(Mat image,Rect partRect){
		

	//Mat imagePart;
	Mat imagePart(partRect.height,partRect.width,CV_8UC3);

    Vec3b color;
	//printf("x: %d\ny: %d\nheight: %d\nwidth: %d\n",partRect.x,partRect.y,partRect.height,partRect.width);
	for(int i = 0 ; i < imagePart.rows;i++){
		for(int j = 0;j < imagePart.cols;j++){

			color = image.at<Vec3b>(Point(j+partRect.x,i+partRect.y));
			imagePart.at<Vec3b>(Point(j,i)) = color;
		}
	}

	//imshow("Skin Image",outputImage);
	return imagePart;
}

Mat clearImage(){
	Mat image(100,100,CV_8UC3);

	Vec3b color(255,255,255);
	for(int i = 0;i < image.rows;i ++){
		for(int j = 0;j < image.cols;j++){
			image.at<Vec3b>(Point(j,i)) = color;
		}
	}
	return image;
}

Mat searchForMovement(Mat thresholdImage, Mat &cameraFeed,Mat output,bool *moveDetect){
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	int xMax = 0,yMax = 0,xMin = 50000,yMin = 50000;
	bool objectDetected = false,peopleDetect = false;
	Rect rect;
	//these two vectors needed for output of findContours
	vector<Vec4i> hierarchy;
	vector<vector<Point>> contours;
	Mat temp;
	thresholdImage.copyTo(temp);
	
	findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_TC89_KCOS );
	
	//if contours vector is not empty, we have found some objects
	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	// retrieves external contours
	
	//if contours vector is not empty, we have found some objects
	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	if(objectDetected){
		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		vector<Point> largerContour;
		int area = 0;
		//this code get the max and minimal points at the contours

		for(int i = 0;i < contours.size();i++){	
			if(contourArea(contours[i]) > 1000){
				for(int j = 0;j < contours[i].size();j++){	
					if(area < contourArea(contours[i])){
						largerContour = contours[i];
						area = contourArea(contours[i]);
						peopleDetect = true;
					}
				}
			}
		}

		for(int i = 0;i < largerContour.size();i++){
			
			if(xMax < largerContour[i].x){
				xMax = largerContour[i].x;
			}

			if(xMin > largerContour[i].x){
				xMin = largerContour[i].x;
			}

			if(yMax < largerContour[i].y){
				yMax = largerContour[i].y;
			}

			if(yMin > largerContour[i].y){
				yMin = largerContour[i].y;
			}
		}

		rect.x = xMin;
		rect.y = yMin;
		rect.height = yMax - yMin;
		rect.width = xMax - xMin;

		

		//printf("%d\n",largestContourVec[0].size());
		//printf("Xmax: %d\nXmin: %d\nYMax: %d\nYMin: %d\n",xMax,xMin,yMax,yMin);
		line(cameraFeed,Point(xMin,yMax),Point(xMax,yMax),Scalar(0,0,255),2);
		line(cameraFeed,Point(xMax,yMax),Point(xMax,yMin),Scalar(0,0,255),2);
		line(cameraFeed,Point(xMax,yMin),Point(xMin,yMin),Scalar(0,0,255),2);
		line(cameraFeed,Point(xMin,yMin),Point(xMin,yMax),Scalar(0,0,255),2);
		
		//update the objects positions by changing the 'theObject' array values
	}
	
	if(peopleDetect){
		*moveDetect = true;
		return getImagePart(cameraFeed,rect);
	}
	*moveDetect = false;
	return output;
	//make some temp x and y variables so we dont have to type out so much
	//int x = theObject[0];
	//int y = theObject[1];
	
	//draw some crosshairs around the object
	
	//circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
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
    Mat cameraFeed,skinMat,fore,movement;
    skindetector mySkinDetector;
    vector<vector<Point> > contours,bContours;
	vector<Vec4i> hierarchy,bHierarchy;
	cv:RNG rng;	
	KalmanFilter KF(4, 2, 0);
	bool flag = false,moveDetect = false;
	int numberOfPeople = 0;
	Rect window;
	Rect partRect;
	movement = clearImage();

	int windowHeight = 320, windowWidth = 400;
	
	// background detection properties
	cv::BackgroundSubtractorMOG2 bg;
    bg.nmixtures = 3;
    bg.bShadowDetection = false;

	
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

		capture.read(cameraFeed);

		bg.operator ()(cameraFeed,fore);
		cv::erode(fore,fore,element);
		cv::dilate(fore,fore,element);	

		movement = searchForMovement(fore,cameraFeed,movement,&moveDetect);
		
		if(!moveDetect)
			movement = clearImage();

		skinMat = mySkinDetector.getSkin(movement);

		cv::erode(skinMat, skinMat, element);
		cv::dilate(skinMat, skinMat, element);
		cv::findContours(skinMat,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0)); //Get Skin contours
	
	
		
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
			//drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
		}
 
		if(mc.inside(window) && flag == false){
			numberOfPeople++;
			flag = true;
			printf("%d ",numberOfPeople);
		}

		if(!mc.inside(window))
			flag = false;
	    
		imshow("skinDetection",skinMat);
		imshow("Original Image",cameraFeed);
        waitKey(30);
    }
    return 0;
}

