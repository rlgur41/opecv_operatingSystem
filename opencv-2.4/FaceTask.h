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

using namespace cv;
using namespace std;


class FaceTask
{
public:
	FaceTask();

	void data_list_read(const string& file_name, vector<Mat>& image_list, vector<int>& label_list);
	void ImageTrainner();
	void WindowsFaceGate();
	void ShowIMG2LBP();
	Mat image2LBP(Mat src);

	~FaceTask();

};

