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
	int x1 = (cv::boundingRect(cv::Mat(contour1)).br()+ cv::boundingRect(cv::Mat(contour1)).tl()).x/2;
	int x2 = (cv::boundingRect(cv::Mat(contour2)).br() + cv::boundingRect(cv::Mat(contour2)).tl()).x / 2;
	return (x1 < x2); // 左的在前
}
class PressureData
{
public:
	double Mx, My, force, Fx, Fy;
	PressureData();
	PressureData(double Mx, double My, double force);
	PressureData& operator=(const PressureData& pData)
	{
		// 处理自我赋值
		if (this == &pData) return *this;
		Mx = pData.Mx;
		My = pData.My;
		force = pData.force;
		Fx = pData.Fx;
		Fy = pData.Fy;
		// 处理链式赋值
		return *this;
	}
	double* retriveData()
	{
		double* data = new double[5];
		data[0] = Mx; data[1] = My; data[2] = force; data[3] = Fx; data[4] = Fy;
		return data;
	}
	friend void swap(PressureData& p1, PressureData& p2)
	{
		swap(p1.Mx, p2.Mx);
		swap(p1.My, p2.My);
		swap(p1.force, p2.force);
		swap(p1.Fx, p2.Fx);
		swap(p1.Fy, p2.Fy);
	}
	void printData()
	{
		printf("Mx: %f\tMy: %f\t Force: %f\tFx: %f\tFy: %f\n", Mx, My, force, Fx, Fy);
	}
};
PressureData::PressureData()
{
	Mx = My = force = Fx = Fy = 0;
}
PressureData::PressureData(double _Mx, double _My, double _force)
{
	Mx = _Mx;
	My = _My;
	force = _force;
	Fx = Mx / _force;
	Fy = My / _force;
}
static Scalar leftScalar(-1, -1, -1, -1), rightScalar(-1, -1, -1, -1);
static int leftGaitCnt = 0;
static int rightGaitCnt = 0;
inline Scalar bottomPoint(Mat m)
{
	Scalar pos(-1, -1, -1, -1);
	for (int y = 0; y < m.rows; y++)
	{
		const uchar* inData = m.ptr<uchar>(y);
		for (int x = 0; x < m.cols; x++)
		{
			if (inData[x])
			{
				pos[0] = x;
				pos[1] = y;
			}
		}
	}
	for (int y = m.rows-1; 0<=y ; y--)
	{
		const uchar* inData = m.ptr<uchar>(y);
		for (int x = m.cols-1; 0<=x; x--)
		{
			if (inData[x])
			{
				pos[2] = x;
				pos[3] = y;
			}
		}
	}
	return pos; 
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
	/*Mat m = Mat::zeros(Size(3, 3), CV_8UC1);
	m.at<uchar>(1, 1) = 1;
	m.at<uchar>(2, 2) = 17;
	copyMakeBorder(m, m, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
	cout << "moment x: " << moments(m).m10 << endl;
	cout << "moment y: " << moments(m).m01 << endl;*/
	
	Mat dilateEle = getStructuringElement(cv::MORPH_DILATE, cv::Size(3, 9));
	
	while (true)
	{
		PressureData lData;
		PressureData rData;
		auto start = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if (frame.empty()) break;
		Mat grey;
		frame *= 5;
		copyMakeBorder(frame, frame, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
		if (frame.channels() == 3) cvtColor(frame, grey, COLOR_BGR2GRAY);
		Mat binary = grey.clone();
		threshold(binary, binary, 0, 255, THRESH_BINARY);
		dilate(binary, binary, dilateEle, Point(-1,-1), 2);
		vector<vector<Point2i>> contours, leftContour, rightContour;
		findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		if (contours.size()<1)
		{
			cout << "contours number invalid" << endl;
			leftGaitCnt = 0;
			rightGaitCnt = 0;
			continue;
		}
		if (2 < contours.size()) break; // serious error, 之后加入报错机制
		double output[6] = { 0,0,0,0,0,0 };
		Mat leftMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat rightMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat left = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat right = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		std::sort(contours.begin(), contours.end(), compareContourX);
		if (contours.size() < 2)
		{
			Rect r = boundingRect(Mat(contours[0]));
			Point br = (r.br()+r.tl())/2;
			if ( br.x< frame.cols/2)
			{
				drawContours(leftMask, contours, -1, Scalar(255), -1); 
				grey.copyTo(left, leftMask);
				Moments lM = moments(left);
				lData = PressureData(lM.m10, lM.m01, sum(left)[0]);
				if (frame.cols / 2 <lData.Fx) swap(lData, rData);
			}
			else
			{
				drawContours(rightMask, contours, -1, Scalar(255), -1);
				grey.copyTo(right, rightMask);
				Moments rM = moments(right);
				rData = PressureData(rM.m10, rM.m01, sum(right)[0]);
				if (rData.Fx < frame.cols / 2) swap(lData, rData);
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
			lData = PressureData(lM.m10, lM.m01, sum(left)[0]);
			Moments rM = moments(right);
			rData = PressureData(rM.m10, rM.m01, sum(right)[0]);
			if (lData.Fx < rData.Fx) swap(lData, rData);
		}
		if (lData.force)
		{
			leftGaitCnt++;
		}
		else
		{
			leftGaitCnt = 0;
		}
		if (rData.force)
		{
			rightGaitCnt++;
		}
		else
		{
			rightGaitCnt = 0;
		}
		if (leftGaitCnt == 1)
		{
			leftScalar = bottomPoint(left)[0];
		}
		if (rightGaitCnt == 1)
		{
			rightScalar = bottomPoint(right);
		}
		if (leftGaitCnt)
		{
			circle(frame, Point(leftScalar[0], leftScalar[1]), 1,Scalar(255, 0, 0), 1);
			circle(frame, Point(leftScalar[2], leftScalar[3]), 1, Scalar(255, 0, 0), 1);
		}
		if (rightGaitCnt)
		{
			circle(frame, Point(rightScalar[0], rightScalar[1]), 1, Scalar(255, 0, 0), 1);
			circle(frame, Point(rightScalar[2], rightScalar[3]), 1, Scalar(255, 0, 0), 1);
		}
		cout << "Left" << endl;
		lData.printData();
		cout << "right" << endl;
		rData.printData();
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end- start;
		std::cout << "Time on export gait data: " << time.count() << std::endl;
		drawContours(frame, contours, -1, Scalar(0,0,255));
		// 水平翻转，现在的视角与面向行走者时看到的视角一样
		flip(frame, frame, 1);
		flip(left, left, 1);
		flip(right, right, 1);
		imshow("Frame", frame);
		imshow("Binary", binary);
		imshow("leftMask", leftMask);
		imshow("rightMask", rightMask);
		imshow("left", left);
		imshow("right", right);
		waitKey(0);
		if (waitKey(1) == 27) break;
	}
}