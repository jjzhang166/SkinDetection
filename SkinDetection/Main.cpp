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

//Define the standard window size
const static int windowHeight = 288, windowWidth = 352;

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
	vector<Vec4i> hierarchy;
	vector<vector<Point>> contours;
	Mat temp;
	thresholdImage.copyTo(temp);
	
	findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_TC89_KCOS );
	
	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	if(objectDetected){

		vector<Point> largerContour;
		int area = 0;

		for(int i = 0;i < (int)contours.size();i++){	
			if(contourArea(contours[i]) > 1000){
				for(int j = 0;j < (int)contours[i].size();j++){	
					if(area < contourArea(contours[i])){
						largerContour = contours[i];
						area = (int)contourArea(contours[i]);
						peopleDetect = true;
					}
				}
			}
		}

		for(int i = 0;i < (int)largerContour.size();i++){
			
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

Mat imageConvert(Mat bgr_image){
    cv::Mat lab_image;

		cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

		// Extract the L channel
		std::vector<cv::Mat> lab_planes(4);
		cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

		// apply the CLAHE algorithm to the L channel
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(3);
		cv::Mat dst;
		clahe->apply(lab_planes[0], dst);

		// Merge the the color planes back into an Lab image
		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

	   // convert back to RGB
	   cv::Mat image_clahe;
	   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	   return image_clahe;
}

void initRectSize(int x, int y, int width, int height,Rect *rect){
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

int main(int argc, const char *argv[]) {
    VideoCapture capture;
    Mat cameraFeed,leftSkinMat,rightSkinMat,leftFrame,rightFrame,skinMat;
    vector<vector<Point> > leftContours,rightContours;
	vector<Point2f> leftDirection(5),rightDirection(5);
	vector<Vec4i> leftHierarchy,rightHierarchy;
	KalmanFilter KF(4, 2, 0);
	Rect leftR,rightR,window,partRect;
	bool lFlag = false,rFlag = false;
	int numberOfPeople = 0;
	int lTemp = 0,rTemp = 0;
	int lX = 0,lDx = 0,rX = 0,rDx = 0;
	

	
	//Init Kalman filter configurations
	initFunctions(KF);
	// init camera capture configuration
	initCapture(&capture);
	// init rect sizes
	initRectSize(0,0,windowWidth,windowHeight,&window); //Used to search the skins in the window.
	initRectSize(0,0,50,windowHeight,&leftR);//Used to detect people in the left of the window.
	initRectSize(windowWidth - 50,0,50,windowHeight,&rightR);//Used to detect people in the left of the window.

	while(1){
		double leftContourArea = 0;
		double rightContourArea = 0;

		capture.read(cameraFeed);
	
		//meraFeed = imageConvert(cameraFeed);

		leftFrame = getImagePart(cameraFeed,leftR);
		rightFrame = getImagePart(cameraFeed,rightR);

		getSkinMat(leftFrame,&leftSkinMat);
		getSkinMat(rightFrame,&rightSkinMat);

	    int j = -1;
		int k = -1;

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

