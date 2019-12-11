#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}
// 比较两个
bool compareContourX(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	int x1 = cv::boundingRect(cv::Mat(contour1)).x;
	int x2 = cv::boundingRect(cv::Mat(contour2)).x;
	return (x1 < x2); // 左的在前
}
int main()
{
	VideoCapture cap("out.avi");
	Mat frame;
	namedWindow("Frame", cv::WINDOW_NORMAL);
	namedWindow("Binary", cv::WINDOW_NORMAL);
	namedWindow("leftMask", cv::WINDOW_NORMAL);
	namedWindow("rightMask", cv::WINDOW_NORMAL);
	namedWindow("left", cv::WINDOW_NORMAL);
	namedWindow("right", cv::WINDOW_NORMAL);
	Mat m = Mat::zeros(Size(3, 3), CV_8UC1);
	m.at<uchar>(1, 1) = 1;
	m.at<uchar>(2, 2) = 17;
	copyMakeBorder(m, m, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
	cout << "moment x: " << moments(m).m10 << endl;
	cout << "moment y: " << moments(m).m01 << endl;
	Mat dilateEle = getStructuringElement(cv::MORPH_DILATE, cv::Size(5, 5));
	while (true)
	{
		auto start = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if (frame.empty()) break;
		Mat grey;
		copyMakeBorder(frame, frame, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
		if (frame.channels() == 3) cvtColor(frame, grey, COLOR_BGR2GRAY);
		Mat binary = grey.clone();
		
		threshold(binary, binary, 0, 255, THRESH_BINARY);
		dilate(binary, binary, dilateEle, Point(-1,-1), 3);
		vector<vector<Point2i>> contours, leftContour, rightContour;
		findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		if (2 < contours.size() || contours.size()<1)
		{
			cout << "contours number invalid" << endl;
			continue;
		}
		double output[6] = { 0,0,0,0,0,0 };
		Mat leftMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat rightMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat left = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat right = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		std::sort(contours.begin(), contours.end(), compareContourX);
		if (contours.size() < 2)
		{
			if (boundingRect(Mat(contours[0])).x < frame.cols/3)
			{
				drawContours(leftMask, contours, -1, Scalar(255), -1); 
				grey.copyTo(left, leftMask);
				Moments lM = moments(left);
				output[0] = lM.m10; // x moment
				output[1] = lM.m01; // y moment
				output[2] = sum(left)[0];
			}
			else
			{
				drawContours(rightMask, contours, -1, Scalar(255), -1);
				grey.copyTo(right, rightMask);
				Moments rM = moments(right);
				output[3] = rM.m10; // x
				output[4] = rM.m01; // y
				output[5] = sum(right)[0];
			}

		}
		else
		{
			leftContour.push_back(contours[0]);
			rightContour.push_back(contours[1]);
			drawContours(leftMask, leftContour, -1, Scalar(255), -1);
			drawContours(rightMask, rightContour, -1, Scalar(255),-1);
			grey.copyTo(left, leftMask);
			grey.copyTo(right, rightMask);
			Moments lM = moments(left);
			output[0] = lM.m10; // x
			output[1] = lM.m01; // y
			output[2] = sum(left)[0];
			Moments rM = moments(right);
			output[3] = rM.m10; // x
			output[4] = rM.m01; // y
			output[5] = sum(right)[0];
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end- start;
		std::cout << "Time on export gait data: " << time.count() << std::endl;

		cout << "left x"<<output[0]<<" left y:"<< output[1] << endl;
		cout << "right x" << output[3] << " right y:" << output[4] << endl;
		cout << "left force" << output[2] << " right force " << output[5] << endl;
		drawContours(frame, contours, -1, Scalar(0,0,255));
		imshow("Frame", frame);
		imshow("Binary", binary);
		imshow("leftMask", leftMask);
		imshow("rightMask", rightMask);
		imshow("left", left);
		imshow("right", right);
		if (waitKey(60) == 27) break;
	}
}