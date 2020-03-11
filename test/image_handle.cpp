#include "image_handle.hpp"

static uchar find_min(uchar n1,uchar n2,uchar n3) 
{      
    return n1<n2?(n1<n3?n1:n3):(n2<n3?n2:n3);
}

static uchar find_max(uchar n1,uchar n2,uchar n3) 
{      
    return n1>n2?(n1>n3?n1:n3):(n2>n3?n2:n3);
}
static uchar find_med(uchar n1,uchar n2,uchar n3) 
{
    uchar temp;

    if (n1 > n2)
    {
        temp = n1;
        n1 = n2;
        n2 = temp;
    }
    if (n2 > n3)
    {
        temp = n2;
        n2 = n3;
        n3 = temp;
    }
    return n1 > n2?n1:n2;
}

static void sort(uchar* n1,uchar* n2,uchar* n3) 
{
    uchar temp;

    if (*n1 > *n2)
    {
        temp = *n2;
        *n2 = *n1;
        *n1 = temp;
    }
    if (*n2 > *n3)
    {
        temp = *n3;
        *n3 = *n2;
        *n2 = temp;
    }
    if (*n1 > *n2)
    {
        temp = *n2;
        *n2 = *n1;
        *n1 = temp;
    }
}

static uchar Median(uchar *kenel,uchar row_input[3],int k) 
{   
    uchar temp;
    int i ,j;
    uchar min,med,max;

    /* bubble sort*/
    for (i = 0; i < 3; i++)
    {   
        for (j = 0; j < 3 - i -1; j++)
        { 
            if (row_input[j] > row_input[j + 1])
            {
                temp = row_input[j];
                row_input[j] = row_input[j + 1];
                row_input[j + 1] = temp;
            }
        }
        *(kenel+3*(j+1)+k) = row_input[j + 1];
    }
    *(kenel+k) = row_input[0];
  
    min = find_min(*(kenel+6),*(kenel+7),*(kenel+8));
    med = find_med(*(kenel+3),*(kenel+4),*(kenel+5));
    max = find_max(*(kenel+0),*(kenel+1),*(kenel+2));
    return find_med(min,med,max);
    
#if 0
    /* bubble sort*/
    for (int i = 0; i < 9/2 + 1; i++)
    {   
        for (int j = i; j < 9 - 1 - i; j++)
        { 
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr[4];
    #endif
}

//直方图均衡
void histogram_equalization (const Mat src, Mat& dst)
{
    int gray[256] = { 0 };  
    double gray_distribution[256] = { 0 };  
    int gray_equal[256] = { 0 };
    int gray_sum = src.cols * src.rows;
    Mat yuvImg;

    if (!src.data || !dst.data)
        return;

    fcv::cvtColor(src, yuvImg, CV_BGR2YUV_I420);

    for (int i = 0; i < gray_sum; i++)
    {
            gray[yuvImg.data[i]]++;
        
    }
    gray_distribution[0] = ((double)gray[0] / gray_sum);

    for (int i = 1; i < 256; i++)
    {
        gray_distribution[i] = gray_distribution[i-1] + (double)gray[i] / gray_sum;
        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
    }

    for (int i = 0; i < gray_sum; i++)
    {
        yuvImg.data[i] = gray_equal[yuvImg.data[i]];
    }

    fcv::cvtColor(yuvImg, dst, CV_YUV2BGR_I420);

} 

#if 0
static uchar MyMedian(uchar kenel) {
	int len = sizeof(kenel);
	uchar temp;
	for (int i = 0; i < len/2 + 1; i++){
		for (int j = 0; j < length - 1; j++){
			if (kenel[j] < kenel[j+1]){
				temp = kenel[j];
				kenel[j] = kenel[j + 1];
				kenel[j + 1] = temp;
			}
		}
	}
	return kenel[4];
}
#endif

//中值过滤
void median_smooth_flitering(const Mat src, Mat &dst) 
{
    //uchar kenel[3][3];
	uchar kenel[9];
	int len = sizeof(kenel);
	uchar temp;

	uchar input_col[3];
    Mat yuvImg;
    int gray_sum = src.cols * src.rows; 

    if (!src.data || !dst.data)
        return;

    uchar* pYuvBuf = new unsigned char[gray_sum];
    fcv::cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    memcpy( pYuvBuf, yuvImg.data,gray_sum*sizeof(unsigned char)); 

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0;  j< src.cols; j++){
			if (i > 0 && i < (src.rows - 1) && j > 0 && j < (src.cols - 1)) {
				//取出kenel
				kenel[0] = pYuvBuf[(i - 1) * src.cols + j-1];
				kenel[1] = pYuvBuf[(i - 1) * src.cols + j];
				kenel[2] = pYuvBuf[(i - 1) * src.cols + j + 1];
				kenel[3] = pYuvBuf[i * src.cols + j-1];
				kenel[4] = pYuvBuf[i * src.cols + j];
				kenel[5] = pYuvBuf[i * src.cols + j + 1];
				kenel[6] = pYuvBuf[(i + 1) * src.cols + j -1];
				kenel[7] = pYuvBuf[(i + 1) * src.cols + j];
				kenel[8] = pYuvBuf[(i + 1) * src.cols + j + 1];


				for (int i = 0; i < len / 2 + 1; i++) {
					for (int j = 0; j < len - 1; j++) {
						if (kenel[j] < kenel[j + 1]) {
							temp = kenel[j];
							kenel[j] = kenel[j + 1];
							kenel[j + 1] = temp;
						}
					}
				}

				yuvImg.data[i * src.cols + j] = kenel[4];
			}
			else
			{
				yuvImg.data[i * src.cols + j] = pYuvBuf[i * src.cols + j];
			}
		}
	}

#if 0
    for(int i=0;i<src.rows;++i)     
    {
        if ( i > 0 && i < (src.rows - 1))
        {
            kenel[0][0] = pYuvBuf[(i-1)*src.cols];
            kenel[0][1] = pYuvBuf[(i-1)*src.cols+1];
            kenel[0][2] = pYuvBuf[(i-1)*src.cols+2];
            kenel[1][0] = pYuvBuf[i*src.cols];
            kenel[1][1] = pYuvBuf[i*src.cols+1];
            kenel[1][2] = pYuvBuf[i*src.cols+2];
            kenel[2][0] = pYuvBuf[(i+1)*src.cols];
            kenel[2][1] = pYuvBuf[(i+1)*src.cols+1];
            kenel[2][2] = pYuvBuf[(i+1)*src.cols+2];
            sort(&kenel[0][0],&kenel[1][0],&kenel[2][0]);
            sort(&kenel[0][1],&kenel[1][1],&kenel[2][1]);
            sort(&kenel[0][2],&kenel[1][2],&kenel[2][2]);
        }
        for (int j=0; j < src.cols; ++j) 
        {          
            if (i > 0 && i < (src.rows - 1) && j > 0 && j < (src.cols - 1)) 
            {
                input_col[0] = pYuvBuf[(i-1)*src.cols+j+1];
                input_col[1] = pYuvBuf[i*src.cols+j+1];
                input_col[2] = pYuvBuf[(i+1)*src.cols+j+1];

                yuvImg.data[i*src.cols+j] = Median(&kenel[0][0],input_col,(j+1)%3);           
            }  
        }
    }
#endif
    fcv::cvtColor(yuvImg, dst, CV_YUV2BGR_I420);
    delete pYuvBuf;
}

//边缘检测
void edge_extraction(const Mat src, Mat &dst)
{
    Mat yuvImg;
    int gray_sum = src.cols * src.rows;
    
    if (!src.data || !dst.data)
        return;

    uchar* pYuvBuf = new unsigned char[gray_sum];
    fcv::cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    memcpy( pYuvBuf, yuvImg.data,gray_sum*sizeof(unsigned char)); 
    /*sharpen_flitering Template 3x3:
                              0 1 0
                              1 -4 1
                              0 1 0   
       */
    for(int i= 0; i<src.rows; ++i)
    {
        for(int j= 0; j<src.cols; ++j)
        {   
            if(i > 0 && i < (src.rows - 1) && j > 0 && j < (src.cols - 1))
            {
                yuvImg.data[i*src.cols+j] = cv::saturate_cast<uchar>(-4*pYuvBuf[i*src.cols+j]+pYuvBuf[(i-1)*src.cols+j]+pYuvBuf[(i+1)*src.cols+j]+pYuvBuf[i*src.cols+j-1]+pYuvBuf[i*src.cols+j+1]);
            }
        }
    }
    fcv::cvtColor(yuvImg, dst, CV_YUV2BGR_I420);
    delete pYuvBuf;
}

void sharpen_flitering(const Mat src, Mat &dst)
{
    Mat yuvImg;
    int gray_sum = src.cols * src.rows;

    if (!src.data || !dst.data)
        return;

    uchar* pYuvBuf = new unsigned char[gray_sum];
    fcv::cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    memcpy( pYuvBuf, yuvImg.data,gray_sum*sizeof(unsigned char));
    /*sharpen_flitering Template 3x3:
                              0 -1 0
                             -1 5 -1
                              0 -1 0   
       */
    for(int i= 0; i<src.rows; ++i)
    {
        for(int j= 0; j<src.cols; ++j)
        {
            if(i > 0 && i < (src.rows - 1) && j > 0 && j < (src.cols - 1))
            {
                yuvImg.data[i*src.cols+j] = cv::saturate_cast<uchar>(5*pYuvBuf[i*src.cols+j]-pYuvBuf[(i-1)*src.cols+j]-pYuvBuf[(i+1)*src.cols+j]-pYuvBuf[i*src.cols+j-1]-pYuvBuf[i*src.cols+j+1]);
            }    
        }
    }
    fcv::cvtColor(yuvImg, dst, CV_YUV2BGR_I420);
    delete pYuvBuf;
}

void bilinear_interpolation(const Mat src, Mat &dst)
{
    if (!src.data || !dst.data)
    {
        return;
    }

    int row = MAGNIFICATION*src.rows, col = MAGNIFICATION*src.cols;
    dst.create(cv::Size(RGA_ALIGN(col, 16), RGA_ALIGN(row, 16)), CV_8UC3);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			int fi = (i + 0.0) / MAGNIFICATION + 0.5;
			int fj = (j + 0.0) / MAGNIFICATION + 0.5;
			dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(fi, fj)[0];
			dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(fi, fj)[1];
			dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(fi, fj)[2];
		}		
	}

#if 0
	for (int i = 0; i < row; ++i) {
		float x = (i + 0.5) / MAGNIFICATION - 0.5;
		int fx = (int)x;

		x -= fx;

		short x1 = 1.f - x;
		short x2 = 1 - x1;
		for (int j = 0; j < col; ++j) {
			float y = (j + 0.5) / MAGNIFICATION - 0.5;
			int fy = (int)y;
			y -= fy;
			short y1 = 1.f - y;
			short y2 = 1 - y1;
			dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(fx, fy)[0] * x1 * y1 + src.at<cv::Vec3b>(fx + 1, fy)[0] * x2 * y1
				+ src.at<cv::Vec3b>(fx, fy + 1)[0] * x1 * y2 + src.at<cv::Vec3b>(fx + 1, fy + 1)[0] * x2 * y2;
			dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(fx, fy)[1] * x1 * y1 + src.at<cv::Vec3b>(fx + 1, fy)[1] * x2 * y1
				+ src.at<cv::Vec3b>(fx, fy + 1)[1] * x1 * y2 + src.at<cv::Vec3b>(fx + 1, fy + 1)[1] * x2 * y2;
			dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(fx, fy)[2] * x1 * y1 + src.at<cv::Vec3b>(fx + 1, fy)[2] * x2 * y1
				+ src.at<cv::Vec3b>(fx, fy + 1)[2] * x1 * y2 + src.at<cv::Vec3b>(fx + 1, fy + 1)[2] * x2 * y2;
		}
	}
#endif // 0


}

static uchar average(const Mat &img, int x_min,int x_max,int y_min,int y_max,int k)
{
    int count = (x_max- x_min + 1) * (y_max - y_min + 1);
    int sum = 0;
    for (int i = x_min; i <= x_max; i++)
    {
        for (int j = y_min; j <= y_max; j++)
        {
            sum += img.at<cv::Vec3b>(i, j)[k];
        }
    }
    return sum/count;
}

void scale_part_average (const Mat src, Mat &dst)
{

    if (!src.data || !dst.data)
    {
        return;
    }
    int rows = static_cast<int>(src.rows * REDUCTION);
    int cols = static_cast<int>(src.cols * REDUCTION);
    dst.create(cv::Size(RGA_ALIGN(cols, 16), RGA_ALIGN(rows, 16)), CV_8UC3);

    int lastRow = 0;
    int lastCol = 0;

    for (int i = 0; i < rows; i++) {
        int row = static_cast<int>((i + 1) / REDUCTION + 0.5) - 1;

        for (int j = 0; j < cols; j++) {
            int col = static_cast<int>((j + 1) / REDUCTION + 0.5) - 1;

            dst.at<cv::Vec3b>(i, j)[0] = average(src, lastRow, row, lastCol, col,0);
            dst.at<cv::Vec3b>(i, j)[1] = average(src, lastRow, row, lastCol, col,1);
            dst.at<cv::Vec3b>(i, j)[2] = average(src, lastRow, row, lastCol, col,2);
            lastCol = col + 1; 
        }
        lastCol = 0; 
        lastRow = row + 1; 
    }
}

void relief(const Mat src, Mat &dst)
{
    Mat yuvImg;
    int gray_sum = src.cols * src.rows;

    if (!src.data || !dst.data)
        return;

    uchar* pYuvBuf = new unsigned char[gray_sum];
    fcv::cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    memcpy( pYuvBuf, yuvImg.data,gray_sum*sizeof(unsigned char));
    for(int i= 0;i<src.rows;++i)
    {
        for(int j= 0; j<src.cols; ++j)
        {
            if(i > 0 && i < (src.rows - 1) && j > 0 && j < (src.cols - 1))
            {
                yuvImg.data[i*src.cols+j] = cv::saturate_cast<uchar>(pYuvBuf[(i-1)*src.cols+j-1]-pYuvBuf[(i+1)*src.cols+j+1]+128);   
            }
        }
    }
    fcv::cvtColor(yuvImg, dst, CV_YUV2BGR_I420);
    delete pYuvBuf;
}

static void rgb_to_gray_to_reverse(const Mat src, Mat &dst1,Mat &dst2)
{
    for (int i = 0; i < src.rows; i++)         
    {                  
        uchar*p = dst1.ptr<uchar>(i);
        uchar*q = dst2.ptr<uchar>(i);
        for(int j = 0; j < src.cols; j++)
        {                            
            *p= (15*src.at<cv::Vec3b>(i, j)[0] + 75*src.at<cv::Vec3b>(i, j)[1] + 38*src.at<cv::Vec3b>(i, j)[2]) >> 7;
            *q = (255 - *p);
            p++;
            q++;
        }         
    }
}

void sketch(const Mat src, Mat &dst)
{
    Mat gray,gray_reverse,gray_end;

    if (!src.data || !dst.data)
        return; 

    gray.create(cv::Size(RGA_ALIGN(src.cols, 16), RGA_ALIGN(src.rows, 16)), CV_8UC1);
    gray_reverse.create(cv::Size(RGA_ALIGN(src.cols, 16), RGA_ALIGN(src.rows, 16)), CV_8UC1);
    gray_end.create(cv::Size(RGA_ALIGN(src.cols, 16), RGA_ALIGN(src.rows, 16)), CV_8UC1);

    rgb_to_gray_to_reverse(src,gray,gray_reverse); 
    GaussianBlur(gray_reverse,gray_reverse,Size(11,11),0);
    
    for (int i = 0; i < src.rows; i++)         
    {                  
        uchar* p1  = gray.ptr<uchar>(i);       
        uchar* p2  = gray_reverse.ptr<uchar>(i);       
        uchar* q  = gray_end.ptr<uchar>(i);      
        for (int j = 0; j < src.cols; j++)     
        {           
            int tmp1=p1[j];         
            int tmp2=p2[j];         
            q[j] =(uchar) min((tmp1+(tmp1*tmp2)/(256-tmp2)),255);       
        }        
    }
    fcv::cvtColor(gray_end,dst,CV_GRAY2BGR);
    //gray.release();
    //gray_reverse.release();
    //gray_end.release();
}

void frozen(const Mat src, Mat& dst){
	if (!src.data || !dst.data)
	{
		return;
	}

	int row = src.rows, col = src.cols;
	dst.create(cv::Size(RGA_ALIGN(col, 16), RGA_ALIGN(row, 16)), CV_8UC3);

	uchar blue, green, red;
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			red = src.at<cv::Vec3b>(i, j)[0];
			green = src.at<cv::Vec3b>(i, j)[1];
			blue = src.at<cv::Vec3b>(i, j)[2];
			dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(std::abs(red - green - blue) * 3 >> 1);
			dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(std::abs(blue - red - green) * 3 >> 1);
			dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(std::abs(green - red - blue) * 3 >> 1);
		}
	}
}

void salt(const Mat src, Mat &dst) {
	//  椒盐噪声需要循环进行反复赋值
	//  循环变量仅用于循环，不用于赋值的移位等功用
	int typeOfNoice = 0;//  定义噪声种类变量，默认为0（椒噪声），1表示盐噪声
	int Crow = 0;
	int Ccol = 0;
	int salt_sum = src.cols * src.rows / 50;
	int count = 0;
	dst = src.clone();
	for (int i = 0; i < salt_sum; i++){
		//  第一个随机数产生函数用于确定椒盐噪声种类
		typeOfNoice = rand() % 2;//  确定输出仅有1和0两个结果  
		//  使用两个随机数产生函数获取被修改的图像矩阵对应位置
		Crow = rand() % src.rows;
		Ccol = rand() % src.cols;
		//  使用at模板函数进入图像矩阵，修改矩阵参数，即加入噪声
		//  如果是3通道图，则处理像素的三元数据（BGR）
		if (src.type() == CV_8UC3) {
			if (typeOfNoice == 0) {
				dst.at<cv::Vec3b>(Crow, Ccol)[0] = 0;
				dst.at<cv::Vec3b>(Crow, Ccol)[1] = 0;
				dst.at<cv::Vec3b>(Crow, Ccol)[2] = 0;
			}
			else if (typeOfNoice == 1) {
				dst.at<cv::Vec3b>(Crow, Ccol)[0] = 255;
				dst.at<cv::Vec3b>(Crow, Ccol)[1] = 255;
				dst.at<cv::Vec3b>(Crow, Ccol)[2] = 255;
			}
		};
	}
}