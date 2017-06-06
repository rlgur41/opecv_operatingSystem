#pragma once
#include "FaceTask.h"

string classifier = "K:/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
string save_path = "D:/Programming/ImageProcessing/opencv-2.4/trainner2.yml";
string sample_path = "D:/Programming/ImageProcessing/opencv-2.4/dataSet/User.1.46.jpg";
string command_file = "D:/Programming/ImageProcessing/opencv-2.4/order.txt";
string learning_data = "D:/Programming/ImageProcessing/opencv-2.4/trainner2.yml";

FaceTask::FaceTask()
{
}


void FaceTask::data_list_read(const string& file_name, vector<Mat>& image_list, vector<int>& label_list)
{

	ifstream file(file_name.c_str(), ifstream::in);

	if (!file) {
		printf("[FAIL] File open Error\n");
		return;
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, ';');
		getline(liness, label);
		if (!path.empty() && !label.empty()) {
			image_list.push_back(imread(path, 0));
			label_list.push_back(atoi(label.c_str()));
		}
	}
}

void FaceTask::ImageTrainner() {

	vector<Mat> image_list;
	vector<int> label_list;


	try {
		data_list_read(command_file, image_list, label_list);
		printf("Image list size : %d\n", image_list.size());
		printf("label list size : %d\n", label_list.size());

	}
	catch (cv::Exception& e) {
		printf("[FAIL] File open error\n");
		return;
	}

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

	//학습 시작 
	model->train(image_list, label_list);
	printf("==================== Trainning start ====================\n");

	//학습 데이터 저장
	model->save(save_path);
	printf("Save location : %s\n", save_path);
}


int FaceTask::WindowsFaceGate(Mat& frame, VideoCapture& cap) {
	bool result = false;
	clock_t limit_time;

	cout << "==================== Start recognition ====================" << endl;

	Ptr<FaceRecognizer>  model = createLBPHFaceRecognizer();
	model->load(learning_data);

	CascadeClassifier face_cascade;

	if (!face_cascade.load(classifier)) {
		cout << " Error loading file" << endl;
		return -2;
	}

	clock_t start = clock();


	Mat testSample = imread(sample_path, 0);
	int img_width = testSample.cols;
	int img_height = testSample.rows;


	while (true)
	{
		cap >> frame;

		vector<Rect> faces;
		Mat graySacleFrame;
		Mat original;
		Mat test_mat;

		if (!frame.empty()) {
			int width = 0;
			int height = 0;

			Rect roi;

			original = frame.clone();
			test_mat = frame.clone();
		
			cvtColor(frame, graySacleFrame, CV_BGR2GRAY);
			equalizeHist(graySacleFrame, graySacleFrame);

			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			for (int i = 0; i < faces.size(); i++)
			{
				Rect face_i = faces[i];
				Mat face_resized;
				Mat face = graySacleFrame(face_i);

				int label = -1;
				double confidence = 0.0;

				roi.x = faces[i].x; roi.width = faces[i].width;
				roi.y = faces[i].y; roi.height = faces[i].height;

				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
				cv::resize(test_mat, test_mat, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
			
				model->predict(face_resized, label, confidence);

				cout << " confidencde " << confidence << endl;

				rectangle(original, face_i, Scalar(255, 0, 0), 1);
				if (confidence < 50.0) {
					if (label == 1) {
						result = true;
						imshow("test", face_resized);
					}
					else {
						result = false;
					}
				}
			}
		
			putText(original, "authentication: " + result,  Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
		
			clock_t end = clock();

			limit_time = (end - start) / (double)CLOCKS_PER_SEC;

			cout << "elapsed time : " << limit_time << endl;

			if (limit_time > 10 && !result) {
				printf("Auth time over\n");
				LockWorkStation();
				return -1;
			}
			else if (result) {
				destroyAllWindows();
				return 1;
			}
			
			imshow("facegate", original);
		}
		if (waitKey(30) >= 0) break;
	}
}

Mat FaceTask::image2LBP(Mat src)
{
	bool value = true;
	Mat Image(src.rows, src.cols, CV_8UC1);
	Mat lbp(src.rows, src.cols, CV_8UC1);

	if (src.channels() == 3)
		cvtColor(src, Image, CV_BGR2GRAY);

	unsigned int center = 0;
	unsigned int center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)
	{
		for (int col = 1; col < Image.cols - 1; col++)
		{
			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;//2

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;//4

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 8;//8

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 16;//16

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 32;//32

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 64;//64

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 128;//128
			lbp.at<uchar>(row, col) = center_lbp;
		}
	}

	return lbp;

}

void FaceTask::ShowIMG2LBP()
{

	Mat img = imread("User.1.39.jpg");


	imshow("test", (img));

	//	imshow("wow", image2LBP(imread("test1.jpg", 1)));
	waitKey(1000000);
}

FaceTask::~FaceTask()
{
}
