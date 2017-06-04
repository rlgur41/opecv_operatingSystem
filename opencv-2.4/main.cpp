

//#include "FaceTask.h"
#include "HandTask.h"


int main(int argc, char** argv)
{
	static Point pos;
	HandTask htask = HandTask();
	VideoCapture cap(0); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"


	double max_area = 0;


	vector<vector<Point>> contours, contoursBlue;
	vector<Vec4i> hierarchy, hierarchyBlue;
	bool flag = false;
	int posX, posY;

	while (true)
	{
		int max_cidx_r = -1;
		int max_cidx_y = -1;

		Mat imgOriginal;
		Mat imgHSV;


		cap >> imgOriginal;


		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded, imgThresholdedBlue;
		Mat imgContour, imgContourBlue;
		//red scalar Scalar(170, 50, 0), Scalar(179, 255, 255)
		inRange(imgHSV, Scalar(100, 150, 0), Scalar(140, 255, 255), imgThresholdedBlue);
		inRange(imgHSV, Scalar(20, 100, 100), Scalar(30, 255, 255), imgThresholded); //Threshold the image
		//yello Scalar(20, 100, 100), Scalar(30, 255, 255)


		imgContour = imgThresholded.clone();
		imgContourBlue = imgThresholdedBlue.clone();

		htask.erode_dilate(imgThresholded, Size(5, 5));
		htask.erode_dilate(imgThresholdedBlue, Size(5, 5));

		int ret_y, ret_b;
		if (-1 != (ret_y = htask.getMaxContoursIdx(imgContour, contours, hierarchy, 0))) {
			Rect rect_y = boundingRect(contours[ret_y]);
			rectangle(imgOriginal, rect_y, Scalar(0, 255, 0), 1, 8, 0);
			pos = htask.getMousePosition(rect_y, 320);
			mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, pos.x, pos.y, 0, 0);
		}
		
		if(-1 != (ret_b = htask.getMaxContoursIdx(imgContourBlue, contoursBlue, hierarchyBlue, 1))) {
			Rect rect_b = boundingRect(contoursBlue[ret_b]);
			rectangle(imgOriginal, rect_b, Scalar(255, 0, 0), 1, 8, 0);
		}

	
		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image
		imshow("ThresholdedBlue Image", imgThresholdedBlue);

		if (waitKey(30) == 27) 
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;

}