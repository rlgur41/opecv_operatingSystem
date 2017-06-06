

//#include "FaceTask.h"
#include "HandTask.h"
#include "FaceTask.h"
int main(int argc, char** argv)
{
	unsigned char protocol = 0x00; // 0000 0000

	static Point pos;

	HandTask htask = HandTask();
	FaceTask ftask = FaceTask();
	Mat imgOriginal;


	vector<vector<Point>> contours, contoursBlue;
	vector<Vec4i> hierarchy, hierarchyBlue;

	VideoCapture cap(0); 

	if (!cap.isOpened()) 
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}


	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	if (ftask.WindowsFaceGate(imgOriginal, cap) != 1) {
		return -1;
	}
	else {
		while (true)
		{
			int max_cidx_r = -1;
			int max_cidx_y = -1;

			cap >> imgOriginal;

			Mat imgHSV;
			cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

			Mat imgThresholded, imgThresholdedBlue;
			Mat imgContour, imgContourBlue;
			//red scalar Scalar(170, 50, 0), Scalar(179, 255, 255)
			inRange(imgHSV, Scalar(100, 150, 0), Scalar(140, 255, 255), imgThresholdedBlue);
			inRange(imgHSV, Scalar(20, 100, 100), Scalar(30, 255, 255), imgThresholded);      //Threshold the image
			//yello Scalar(20, 100, 100), Scalar(30, 255, 255)


			imgContour = imgThresholded.clone();
			imgContourBlue = imgThresholdedBlue.clone();

			htask.erode_dilate(imgThresholded, Size(5, 5));
			htask.erode_dilate(imgThresholdedBlue, Size(5, 5));

			int ret_y, ret_b;
			if (-1 != (ret_y = htask.getMaxContoursIdx(imgContour, contours, hierarchy, 0))) {
				protocol = protocol | 0x01;
				Rect rect_y = boundingRect(contours[ret_y]);
				rectangle(imgOriginal, rect_y, Scalar(0, 255, 0), 1, 8, 0);
				pos = htask.getMousePosition(rect_y, 320);
			}
			else {							// 0000 0001
				protocol = protocol & 0x02; // 0000 0010
			}

			if (-1 != (ret_b = htask.getMaxContoursIdx(imgContourBlue, contoursBlue, hierarchyBlue, 1))) {
				protocol = protocol | 0x02; // 0000 0010
				Rect rect_b = boundingRect(contoursBlue[ret_b]);
				rectangle(imgOriginal, rect_b, Scalar(255, 0, 0), 1, 8, 0);
			}
			else { //0000 0011
				protocol = protocol & 0x01; // 0000 0010
			}

			switch (protocol & 0x0F) {
			case 0x00:
			//	cout << "noting to do" << endl;
				break;
			case 0x01:
				mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, pos.x, pos.y, 0, 0);
				break;
			case 0x02:
				mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN, pos.x, pos.y, 0, 0);
				mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP, pos.x, pos.y, 0, 0);
				break;
			case 0x03:
				mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_RIGHTDOWN, pos.x, pos.y, 0, 0);
				mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_RIGHTUP, pos.x, pos.y, 0, 0);
			}

			imshow("Original", imgOriginal); //show the original image

			if (waitKey(30) == 27)
			{
				cout << "esc key is pressed" << endl;
				break;
			}
		}
	}

	return 0;

}