#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <time.h>
#include <Windows.h>

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

class HandTask
{

public:
	HandTask();

	void erode_dilate(Mat&, Size);
	int getMaxContoursIdx(Mat&, vector<vector<Point>>&,
						  vector<Vec4i>&, int);
	Point getMousePosition(Rect, int);

	~HandTask();
};

