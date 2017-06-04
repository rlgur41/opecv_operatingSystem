#include "HandTask.h"

HandTask::HandTask()
{
}

void HandTask::erode_dilate(Mat& thres, Size size)
{
	erode(thres, thres, getStructuringElement(MORPH_ELLIPSE, size));
	dilate(thres, thres, getStructuringElement(MORPH_ELLIPSE, size));

	dilate(thres, thres, getStructuringElement(MORPH_ELLIPSE, size));
	erode(thres, thres, getStructuringElement(MORPH_ELLIPSE, size));
}
int HandTask::getMaxContoursIdx(Mat& thres, vector<vector<Point>>& contours, vector<Vec4i>& hier, int mode)
{
	char message[7];
	double max_area = 0;
	int max_idx = -1;

	if (!mode) {
		strcpy_s(message, "Yellow");
	}
	else {
		strcpy_s(message, "Blue");
	}

	findContours(thres, contours, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	for (int i = 0; i < contours.size(); i++) {

		double area = contourArea(contours[i], false);

		if (area > max_area) {
			if (area > 1000) {
				cout << "Color : " << message << "Contour size : " << area << endl;
				max_area = area;
				max_idx = i;
			}
		}
	}

	return (max_idx == -1) ? -1 : max_idx;
	
}
Point HandTask::getMousePosition(Rect bound_rect, int weight)
{
	Point pos;

	pos.x = -(bound_rect.x + bound_rect.width) / 2 * weight;
	pos.y = (bound_rect.y + bound_rect.height) / 2 * weight;

	return pos;
}


HandTask::~HandTask()
{
}
