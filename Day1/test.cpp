#include "fastcv.hpp"
#include <stdio.h>
#include "openai_io.hpp"

using namespace fcv;

using namespace std;

int main()
{
	string filename = "a15.jpg";
	Mat src_img,dst1_img,dst2_img,dst3_img,dst4_img;
	Mat dst5_img;
	
	//读取图片
	src_img = fcv::imread(filename); 
	
	//将 源图片 另存为 目标图片 重定义像素(列，行) 列倍数 行倍数 ，cv::INTER_LINEAR表示
	fcv::resize(src_img,dst1_img,fcv::Size(src_img.cols*2,src_img.rows*2),0,0,cv::INTER_LINEAR); 
	//fcv::resize(src_img,dst1_img,fcv::Size(0,0),2,3,cv::INTER_LINEAR);
	
	//将 源图片 另存为 目标图片 CV_BGR2GRAY:从彩图变灰度图，CV_BGR2YUV:从彩图变成另一个格式的彩图
	fcv::cvtColor(src_img,dst2_img,CV_BGR2GRAY);
	//fcv::cvtColor(src_img,dst2_img,CV_BGR2YUV);
	
	//取一个位置为(列，行)的点
	Point p = Point(src_img.cols/2,src_img.rows/2);
	//Point p = Point(0,0);
	
	//复制图片到 目的图片
	src_img.copyTo(dst3_img);
	
	//在 源文件 画一个圆，p为圆心，50为半径，fcv::Scale(R,G,B)为三色值，5为线条厚度，8为线类型，0为shift值
	fcv::circle(dst3_img,p,50,fcv::Scalar(0,0,100),5,8,0);

	//在 源文件 画一个矩形，p1为左上顶点，p2为右下顶点
	Point p1 = Point(src_img.cols / 3, src_img.rows / 3);
	Point p2 = Point(src_img.cols * 2 / 3, src_img.rows * 2 / 3);
	src_img.copyTo(dst4_img);
	fcv::rectangle(dst4_img, p1, p2, fcv::Scalar(0, 0, 100), 5, 8, 0);

	//写入文件
	fcv::imwrite("resize.jpg",dst1_img);
	fcv::imwrite("cvt.jpg",dst2_img);
	fcv::imwrite("circle.jpg",dst3_img);
	fcv::imwrite("rectangle.jpg", dst4_img);
	return 0;
}
