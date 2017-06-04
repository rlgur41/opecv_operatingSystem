#include "FaceTask.h"

string classifier = "K:/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
string save_path = "D:/Programming/ImageProcessing/opencv-2.4/trainner2.yml";
string sample_path = "D:/Programming/ImageProcessing/opencv-2.4/dataSet/User.1.46.jpg";
string command_file = "D:/Programming/ImageProcessing/opencv-2.4/order.txt";
string learning_data = "D:/Programming/ImageProcessing/Face_detection_python/trainner/trainner.yml";
//string learning_data = save_path;

FaceTask::FaceTask()
{
}


void data_list_read(const string& file_name, vector<Mat>& image_list, vector<int>& label_list)
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

void ImageTrainner() {

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


void WindowsFaceGate() {
	bool result = false;
	clock_t limit_time;

	printf("==================== Start recognition ====================");

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createLBPHFaceRecognizer();
	model->load(learning_data);

	Mat testSample = imread(sample_path, 0);
	int img_width = testSample.cols;
	int img_height = testSample.rows;

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)) {
		cout << " Error loading file" << endl;
		return;
	}

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return;
	}

	namedWindow(window, 1);
	long count = 0;

	clock_t start = clock();
	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;
		Mat test_mat;
		Mat cropImg;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()) {

			//clone from original frame
			original = frame.clone();
			test_mat = frame.clone();
			//convert image to gray scale and equalize
		//	cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			//	equalizeHist(graySacleFrame, graySacleFrame);

			//convert image to gray scale and equalize
			cvtColor(frame, graySacleFrame, CV_BGR2GRAY);
			equalizeHist(graySacleFrame, graySacleFrame);

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			Rect roi;

			//person name
			string Pname = "";
			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				roi.x = faces[i].x; roi.width = faces[i].width;
				roi.y = faces[i].y; roi.height = faces[i].height;

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
				cv::resize(test_mat, test_mat, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);
				//recognizing what faces detected
				int label = -1;
				double confidence = 0;
				model->predict(face_resized, label, confidence);


				cout << "label" << label << endl;
				cout << " confidencde " << confidence << endl;
				//drawing green rectagle in recognize face
				rectangle(original, face_i, CV_RGB(255, 0, 0), 1);
				if (confidence < 50.0) {
					if (label == 1) {
						Pname = "Kihyuk";
						result = true;
						imshow("test", face_resized);
					}
					else {
						result = false;
						Pname = "unknown";
					}
				}


				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image

				putText(original, Pname, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			}
			std::string rst_str = std::to_string(result);
			cv::putText(frame, "AUTORIZED: " + rst_str, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);

			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 255), 1.0);
			//display to the winodw
			clock_t end = clock();
			limit_time = (end - start) / (double)CLOCKS_PER_SEC;
			printf("elapsed time : %lf\n", limit_time);

			if (limit_time > 10 && !result) {
				printf("Auth time over\n");
				LockWorkStation();
				exit(1);
			}
			else {
				printf("Success\n");
			}
			cv::imshow(window, original);
		}
		if (waitKey(30) >= 0) break;
	}
}

Mat image2LBP(Mat src)
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

void ShowIMG2LBP()
{

	for (int idx = 1; idx < 6; idx++) {
		string imgName = "row";
		imgName.append(1, idx + '0');
		imgName.append(".jpg");
		cout << imgName << endl;
		Mat rowImg = imread(imgName);

		imshow(imgName, image2LBP(rowImg));
	}

	//	imshow("wow", image2LBP(imread("test1.jpg", 1)));
	waitKey(1000000);
}

FaceTask::~FaceTask()
{
}
