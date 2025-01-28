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

		dilatare = temp; 
	}
	return dilatare;
}

//cat de rotund e 
double calcCircularitate(Mat_<uchar> img) {
	int arie = 0;
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++) {
			if (img(r, c) == 0) {
				arie += 1;
			}

		}
	}
	int perimetru = 0;
	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			if (img(r, c) == 0) {
				if ((r > 0 && img(r - 1, c) != 0) || (c > 0 && img(r, c - 1) != 0) || (c < img.cols - 1 && img(r, c + 1) != 0) || (r < img.rows - 1 && img(r + 1, c) != 0) ||
					(r > 0 && c > 0 && img(r - 1, c - 1) != 0) || (r < img.rows - 1 && c < img.cols - 1 && img(r + 1, c + 1) != 0) || (r > 0 && c < img.cols - 1 && img(r - 1, c + 1) != 0) ||
					(r < img.rows - 1 && c > 0 && img(r + 1, c - 1) != 0)) {
					perimetru++;
				}
			}
		}
	}
	float subtiere = 4 * ((float)arie / (perimetru*perimetru)) * PI;
	printf("factprul de subtiere: %f\n", subtiere);
	return subtiere;
}

//  distanta dintre 2 ochi
bool areEyes(const Rect& r1, const Rect& r2) {
	int dx = abs(r1.x - r2.x);
	int dy = abs(r1.y - r2.y);
	double distance = sqrt(dx * dx + dy * dy);
	printf("%d %d %f  ", dx, dy, distance);
	printf("%d %d %d\n", r1.height , r1.width * 4, r1.width * 20);
	//        axa y                   x min                          x max
	return (dy < r1.height ) && (distance > r1.width * 3) && (distance < r1.width * 20);
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
		//Mat mask1 = Mat::zeros(croppedImage.size(), CV_8UC1);
		for (int i = 0; i < croppedImage.rows; i++) {
			for (int j = 0; j < croppedImage.cols; j++) {
				Vec3b pixel = croppedImage.at<Vec3b>(i, j);
				if (pixel[2] > 150 && pixel[2] > (pixel[1] + pixel[0])) {
					mask.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("masca", mask);
		mask = dilatare(mask, 2);
		imshow("masca_dilatare", mask);
		// bfs conturul fiecarei forme
		vector<vector<Point>> contur;
		Mat visited = Mat::zeros(mask.size(), CV_8UC1);

		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				if (mask.at<uchar>(i, j) == 255 && visited.at<uchar>(i, j) == 0) {
					vector<Point> contour;
					Point start = Point(j, i);
					contour.push_back(start);
					visited.at<uchar>(i, j) = 1;
					for (size_t k = 0; k < contour.size(); k++) {
						Point pt = contour[k];
						for (int ki = -1; ki <= 1; ki++) {
							for (int kj = -1; kj <= 1; kj++) {
								Point neighbor = pt + Point(kj, ki);
								if (isInside(mask, neighbor.y, neighbor.x) && mask.at<uchar>(neighbor.y, neighbor.x) == 255 && visited.at<uchar>(neighbor.y, neighbor.x) == 0) {
									contour.push_back(neighbor);
									visited.at<uchar>(neighbor.y, neighbor.x) = 1;
								}
							}
						}
					}
					contur.push_back(contour);
				}
			}
		}
		vector<pair<double, Rect>> circle_reg={};
		for (int i = 0; i < contur.size(); i++) {
			//boundingRect returns a rectangle that fits around the contour
			Rect bounding_box = boundingRect(contur[i]);
			Mat_<uchar> regiune_de_interes = mask(bounding_box);
			double circularity = calcCircularitate(regiune_de_interes);
			if (circularity > 0.2&& circularity<(double)1.3) {
				// cat de de rotund, regiunea/eye
				circle_reg.push_back({ circularity, bounding_box });
			}
		}
		// pentru sortare sa sorteze in functie de cat de rotund e
		auto comparator = [](const pair<double, Rect>& a, const pair<double, Rect>& b) {
			return a.first > b.first;
		};
		std::sort(circle_reg.begin(), circle_reg.end(), comparator);

		vector<Rect> potentialEyes;
		for (size_t i = 0; i < circle_reg.size(); i++) {
			Rect eye = circle_reg[i].second;
			potentialEyes.push_back(eye);
		}
		int finish = 0;
		for (size_t i = 0; i < potentialEyes.size(); i++) {
			for (size_t j = i + 1; j < potentialEyes.size(); j++) {
				if (finish == 1)
					break;
				if (areEyes(potentialEyes[i], potentialEyes[j])) {
					finish = 1;
					//rectangle(croppedImage, potentialEyes[i], Scalar(0, 255, 0), 2);
					//rectangle(croppedImage, potentialEyes[j], Scalar(0, 255, 0), 2);
					Rect eye1 = potentialEyes[i];
					for (int y = eye1.y; y < eye1.y + eye1.height; y++) {
						for (int x = eye1.x; x < eye1.x + eye1.width; x++) {
							if (mask.at<uchar>(y, x) == 255) {
								Vec3b& pixel = croppedImage.at<Vec3b>(y, x);
								pixel[2] = (pixel[0] + pixel[1]) / 2;
							}
						}
					}
					Rect eye2 = potentialEyes[j];
					for (int y = eye2.y; y < eye2.y + eye2.height; y++) {
						for (int x = eye2.x; x < eye2.x + eye2.width; x++) {
							if (mask.at<uchar>(y, x) == 255) {
								Vec3b& pixel = croppedImage.at<Vec3b>(y, x);
								pixel[2] = (pixel[0] + pixel[1]) / 2;
							}
						}
					}
				}
			}
			if (finish == 1)
				break;
		}
		imshow("Red Eye Detection", img);
		waitKey(0);
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