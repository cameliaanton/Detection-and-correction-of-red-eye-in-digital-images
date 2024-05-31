// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Point startPoint, endPoint;
bool selectRect = false;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}
void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}
void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}
void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}
void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}
void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}
void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}
void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}
void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}


struct MouseParams {
	Rect box;
	bool drawing_box;
	Mat* src;

	MouseParams() : box(Rect(-1, -1, 0, 0)), drawing_box(false), src(nullptr) {}
};

// Callback function for mouse events
void MyCallBackFunc(int event, int x, int y, int flags, void* param) {
	MouseParams* mp = (MouseParams*)param;
	switch (event) {
	case EVENT_LBUTTONDOWN:
		mp->drawing_box = true;
		mp->box = Rect(x, y, 0, 0);
		break;
	case EVENT_MOUSEMOVE:
		if (mp->drawing_box) {
			mp->box.width = x - mp->box.x;
			mp->box.height = y - mp->box.y;
		}
		break;
	case EVENT_LBUTTONUP:
		mp->drawing_box = false;
		if (mp->box.width < 0) {
			mp->box.x += mp->box.width;
			mp->box.width *= -1;
		}
		if (mp->box.height < 0) {
			mp->box.y += mp->box.height;
			mp->box.height *= -1;
		}
		break;
	}
}


bool isInside(const Mat& src, int i, int j) {
	return (i >= 0 && i < src.rows&& j >= 0 && j < src.cols);
}


void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void fillHoles(Mat& mask) {
	Mat temp;
	mask.copyTo(temp);
	floodFill(temp, Point(0, 0), Scalar(255));
	bitwise_not(temp, temp);
	mask = (mask | temp);
}
/*
void project() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		imshow("img", img);
		if (img.empty()) {
			cout << "Image not found: " << fname << endl;
			continue;
		}

		namedWindow("Select Region of Interest");
		MouseParams mp;
		mp.src = &img;
		setMouseCallback("Select Region of Interest", MyCallBackFunc, &mp);

		while (true) {
			Mat temp_image = img.clone();
			if (mp.drawing_box) {
				rectangle(temp_image, mp.box, Scalar(0, 255, 255));
			}
			imshow("Select Region of Interest", temp_image);
			if (waitKey(10) == 27) break; // Esc stops the loop
			if (!mp.drawing_box && mp.box.width > 0 && mp.box.height > 0) {
				break;
			}
		}
		destroyWindow("Select Region of Interest");

		if (mp.box.width > 0 && mp.box.height > 0) {
			Mat croppedImage = img(mp.box);
			Mat gray;
			cvtColor(croppedImage, gray, COLOR_BGR2GRAY);
			CascadeClassifier eye_cascade;
			eye_cascade.load("D:/CTI/an3/sem2/PI/red-eye-detection/OpenCVApplication-VS2017_OCV340_basic/OpenCV/include/opencv2/data/haarcascade_eye.xml");

			vector<Rect> eyes;
			eye_cascade.detectMultiScale(croppedImage, eyes, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (Rect eye : eyes) {
				rectangle(croppedImage, eye, Scalar(0, 255, 0), 2);
			}
			imshow("Cropped Image with Green Eyes", croppedImage);
			//imshow("Image", img);
			for (Rect eye : eyes) {
				Mat eyeRegion = croppedImage(eye);
				vector<Mat> bgr(3);
				
				split(eyeRegion, bgr);
				Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));
				fillHoles(mask);
				dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

				Mat mean = (bgr[0] + bgr[1]) / 2;
				mean.copyTo(bgr[0], mask);
				mean.copyTo(bgr[1], mask);
				mean.copyTo(bgr[2], mask);

				Mat eyeOut;
				merge(bgr, eyeOut);
				eyeOut.copyTo(croppedImage(eye));
			}

			imshow("Cropped Image with Green Eyes", croppedImage);
			waitKey(0);
		}

	}
}
*/



Mat dilatare(const Mat& src, int iteration) {
	int height = src.rows;
	int width = src.cols;
	Mat dilatare = src.clone();

	for (int iter = 0; iter < iteration; iter++) {
		Mat temp = dilatare.clone();  

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				bool setPixel = false;
				//prin vecini 
				for (int ki = -1; ki <= 1; ki++) {
					for (int kj = -1; kj <= 1; kj++) {
						if (isInside(src, i + ki, j + kj) && dilatare.at<uchar>(i + ki, j + kj) == 255) {
							setPixel = true;
							break;
						}
					}
					if (setPixel) break;
				}
				if (setPixel) {
					temp.at<uchar>(i, j) = 255;
				}
			}
		}

		dilatare = temp;  // Update the dilated image for the next iteration
	}
	return dilatare;
}

double calculateCircularity(const vector<Point>& contour) {
	double perimeter = arcLength(contour, true);
	double area = contourArea(contour);
	return 4 * CV_PI * (area / (perimeter * perimeter));
}

void project() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		if (img.empty()) {
			cout << "Image not found: " << fname << endl;
			continue;
		}

		namedWindow("Select Region of Interest");
		MouseParams mp;
		mp.src = &img;
		setMouseCallback("Select Region of Interest", MyCallBackFunc, &mp);

		while (true) {
			Mat temp_image = img.clone();
			if (mp.drawing_box) {
				rectangle(temp_image, mp.box, Scalar(0, 255, 255));
			}
			imshow("Select Region of Interest", temp_image);
			if (waitKey(10) == 27) break; // Esc stops the loop
			if (!mp.drawing_box && mp.box.width > 0 && mp.box.height > 0) {
				break;
			}
		}
		destroyWindow("Select Region of Interest");
		Mat croppedImage = img(mp.box);
		Mat mask = Mat::zeros(croppedImage.size(), CV_8UC1);
		for (int i = 0; i < croppedImage.rows; i++) {
			for (int j = 0; j < croppedImage.cols; j++) {
				Vec3b pixel = croppedImage.at<Vec3b>(i, j);
				if (pixel[2] > 150 && pixel[2] > (pixel[1] + pixel[0])) {
					mask.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("masca", mask);
		//fillHoles(mask);
		//imshow("masca1", mask);
		//dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);
		imshow("masca1", mask);
		//mask = dilataree_8(mask);
		mask = dilatare(mask, 3);

		imshow("masca_dilatare", mask);
		
		for (int i = 0; i < croppedImage.rows; i++) {
			for (int j = 0; j < croppedImage.cols; j++) {
				if (mask.at<uchar>(i, j) == 255) {
					Vec3b& pixel = croppedImage.at<Vec3b>(i, j);
					pixel[2] = pixel[0] = pixel[1] = (pixel[0] + pixel[1]) / 2; // Set blue and green channels to mean
				}
			}
		}
		
		//imshow("img1", croppedImage);
		
		imshow("Red Eye Detection", img);
		waitKey(0);
	}
}
void project1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		Mat imgOut= img.clone();
		if (img.empty()) {
			cout << "Image not found: " << fname << endl;
			continue;
		}
		//imshow("img", img);
		namedWindow("Select Region of Interest");
		MouseParams mp;
		mp.src = &img;
		setMouseCallback("Select Region of Interest", MyCallBackFunc, &mp);

		while (true) {
			Mat temp_image = img.clone();
			if (mp.drawing_box) {
				rectangle(temp_image, mp.box, Scalar(0, 255, 255));
			}
			imshow("Select Region of Interest", temp_image);
			if (waitKey(10) == 27) break; // Esc stops the loop
			if (!mp.drawing_box && mp.box.width > 0 && mp.box.height > 0) {
				break;
			}
		}
		destroyWindow("Select Region of Interest");

		if (mp.box.width > 0 && mp.box.height > 0) {
			Mat croppedImage = img(mp.box);
			Mat gray;
			cvtColor(croppedImage, gray, COLOR_BGR2GRAY);
			CascadeClassifier eye_cascade;
			eye_cascade.load("D:/CTI/an3/sem2/PI/red-eye-detection/OpenCVApplication-VS2017_OCV340_basic/OpenCV/include/opencv2/data/haarcascade_eye.xml");

			vector<Rect> eyes;
			eye_cascade.detectMultiScale(gray, eyes, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (Rect eye : eyes) {
				rectangle(croppedImage, eye, Scalar(0, 255, 0), 2);
			}
			for (Rect& eye : eyes) {
				Mat eyeRegion = croppedImage(eye);
				Mat mask = Mat::zeros(eye.height, eye.width, CV_8U);
				for (int i = 0; i < eye.height; i++) {
					for (int j = 0; j < eye.width; j++) {
						Vec3b pixel = eyeRegion.at<Vec3b>(i, j);
						if (pixel[2] > 150 && pixel[2] > (pixel[1] + pixel[0])) {
							mask.at<uchar>(i, j) = 255;
						}
					}
				}
				imshow("masca", mask);
				fillHoles(mask);
				imshow("masca1", mask);
				dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);
				imshow("masca2", mask);
				for (int i = 0; i < eye.height; i++) {
					for (int j = 0; j < eye.width; j++) {
						if (mask.at<uchar>(i, j) == 255) {
							Vec3b& pixel = eyeRegion.at<Vec3b>(i, j);
							pixel[2]=pixel[0] = pixel[1] = (pixel[0] + pixel[1]) / 2; // Set blue and green channels to mean
						}
					}
				}
				eyeRegion.copyTo(croppedImage(eye));
			}
			croppedImage.copyTo(imgOut(mp.box));

			imshow("Cropped Image with Green Eyes", croppedImage);
			//imshow("Output Image", imgOut); 
			waitKey(0);
		}
	}
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				project();
				break;
		}
	}
	while (op!=0);
	return 0;
}