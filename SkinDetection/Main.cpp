#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

//Define the standard window size
const static int windowHeight = 288, windowWidth = 352;


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
    // Range of skin color values
    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 170;
    Cb_MIN = 75;
    Cb_MAX = 135;
}

skindetector::~skindetector(void)
{
}

Mat skindetector::getSkin(cv::Mat input)
{
    Mat skin;
    //first convert our RGB image to YCrCb
    cvtColor(input,skin,cv::COLOR_BGR2YCrCb);
    
    //filter the image in YCrCb color space
    inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
    return skin;
}

Mat getImagePart(Mat image,Rect partRect){
	
	// Create an image with the size of rect received
	Mat imagePart(partRect.height,partRect.width,CV_8UC3);
	// Temp variable to copy the pixel value
    Vec3b color;

	for(int i = 0 ; i < imagePart.rows;i++){
		for(int j = 0;j < imagePart.cols;j++){
			// Copy the pixel value of the original image, and put it in the new image with new rect
			color = image.at<Vec3b>(Point(j+partRect.x,i+partRect.y));
			imagePart.at<Vec3b>(Point(j,i)) = color;
		}
	}
	return imagePart;
}

void initRectSize(int x, int y, int width, int height,Rect *rect){
	//initialize the rect values
	rect->x = x;
	rect->y = y;
	rect->width = width;
	rect->height = height;
}

void initCapture(VideoCapture* capture){
	capture->open(0);
	capture->set(CV_CAP_PROP_FRAME_WIDTH,windowWidth);
	capture->set(CV_CAP_PROP_FRAME_HEIGHT,windowHeight);
	capture->set(CV_CAP_PROP_HUE,8); //HUE 8
	capture->set(CV_CAP_PROP_SATURATION,93); //saturation 93
}

void getSkinMat(Mat frame,Mat *outImage){
	skindetector mySkinDetector;
	vector<vector<Point>> *contours = new vector<vector<Point>>;
	vector<Vec4i> hierarchy;
	RNG rng;

	int erosion_size = 2;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(erosion_size +1 ,erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    

	Scalar color = Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) );

	*outImage = mySkinDetector.getSkin(frame);

	cv::erode(*outImage, *outImage, element);
	cv::dilate(*outImage, *outImage, element);
	
}

vector<vector<Point>> getContours(Mat image,int* x){
	vector<vector<Point>> contours;
	vector<Vec4i>hierarchy;
	double contourArea = 0;
	// Search the contours in the image and put it in a Matrix of points.
	findContours(image,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0)); //Get Skin contours

		for( int i = 0; i< (int)contours.size(); i++) {
			if(cv::contourArea(contours[i]) > 300){
				if(cv::contourArea(contours[i]) > contourArea){
					contourArea = cv::contourArea(contours[i]);
					*x = i;
				}
				drawContours( image, contours, i,Scalar(127,127,127), 1.5, 1, hierarchy, CV_16SC1, Point() );
			}
		}
		return contours;
}

int main() {
    VideoCapture capture;
    Mat cameraFeed,leftSkinMat,rightSkinMat,leftFrame,rightFrame,skinMat;
    vector<vector<Point> > leftContours,rightContours;
	vector<Point2f> leftDirection(5),rightDirection(5);
	vector<Vec4i> leftHierarchy,rightHierarchy;
	Rect leftR,rightR,window,partRect;

	bool lFlag = false,rFlag = false;
	int numberOfPeople = 0;
	int lTemp = 0,rTemp = 0;
	int lX = 0,lDx = 0,rX = 0,rDx = 0;
	

	
	//Init Kalman filter configurations
	// init camera capture configuration
	initCapture(&capture);
	// init rect sizes
	initRectSize(0,0,windowWidth,windowHeight,&window); //Used to search the skins in the window.
	initRectSize(0,0,50,windowHeight,&leftR);//Used to detect people in the left of the window.
	initRectSize(windowWidth - 50,0,50,windowHeight,&rightR);//Used to detect people in the left of the window.

	while(1){
		
		//initialize the camera
		capture.read(cameraFeed);
	
		//Get the sides frames 
		leftFrame = getImagePart(cameraFeed,leftR);
		rightFrame = getImagePart(cameraFeed,rightR);

		//Get the skins in the rect of before frames.
		getSkinMat(leftFrame,&leftSkinMat);
		getSkinMat(rightFrame,&rightSkinMat);

		//Index of the bigger area contour. Used to count the number of people passed.
	    int j = -1;
		int k = -1;

		//Get the contours created by the skins in the rect.
		leftContours = getContours(leftSkinMat,&j);
		rightContours = getContours(rightSkinMat,&k);
		
		
		Moments lmu,rmu;	
		Point2f lmc = -1,rmc = -1;
		lDx = 0;
		rDx = 0;


		//printf("%d",j);

		if(j < 0)
			lTemp = 0;
		if(k < 0)
			rTemp = 0;


		if(j >= 0 && lTemp < 5){
			lmu = moments( leftContours[j], false );
			lmc = Point2f( lmu.m10/lmu.m00 , lmu.m01/lmu.m00 ); 
			leftDirection[lTemp] = Point2f( lmu.m10/lmu.m00 , lmu.m01/lmu.m00 );
			//drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
			lTemp++;
		}

		if(k >= 0 && rTemp < 5){
			rmu = moments(rightContours[k],false);
			rmc = Point2f( rmu.m10/rmu.m00 , rmu.m01/rmu.m00 ); 
			rightDirection[rTemp] = Point2f( rmu.m10/rmu.m00 , rmu.m01/rmu.m00 );
			//drawCross(cameraFeed, mc, Scalar(255, 0, 0), 5);
			rTemp++;
		}
 

		if(lTemp == 5){
			lX = (int)leftDirection[0].x;
			for(int i = 1;i < (int)leftDirection.size();i++){
				lDx += (int)(lX - leftDirection[i].x); 
			}
		}
		if(rTemp == 5){
			rX = rightDirection[0].x;
			for(int i = 0;i < (int)rightDirection.size();i++){
				rDx += (int)(rX - rightDirection[i].x);
			}
		}

		//printf("ldx: %d jrdx: %d\n",lDx,rDx);

		if(lmc.inside(window) && lFlag == false){
			if(lDx < 0){
				numberOfPeople++;
				lFlag = true;
				printf("%d ",numberOfPeople);
			}
		}
		if(rmc.inside(window) && rFlag == false){
			if(rDx > 0){
				numberOfPeople++;
				rFlag = true;
				printf("%d ",numberOfPeople);
			}
		}

		if(!lmc.inside(window))
			lFlag = false;
		if(!rmc.inside(window))
			rFlag = false;
		
	

		getSkinMat(cameraFeed,&skinMat);

		imshow("right",rightSkinMat);
		imshow("left",leftSkinMat);
		imshow("Original Image",skinMat);
        waitKey(30);
    }
    return 0;
}

