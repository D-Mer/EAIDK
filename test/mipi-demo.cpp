/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: minglu@openailab.com
 *
 */
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include <sys/time.h>
 //#include "common.hpp"
#include "mipi_cam.hpp"
#include "image_handle.hpp"

#include "fastcv.hpp"

extern "C" {
#include <rockchip/rockchip_rga.h>
}
#include <rockchip/rockchip_isp.h>

using namespace fcv;
using namespace std;

#define MAGNIFICATION  1.5
#define REDUCTION 0.5

#define DEF_MODEL "models/MobileNetSSD_deploy.tmfile"
#define DEF_IMAGE "ssd_dog.jpg"

const string wintitle = "mipi-camera";
const struct scene_processing scene_processing_arr[SCENE_MAX] = {
                                {"histogram-equalization",&histogram_equalization},
                                {"smooth-filtering(median)",&median_smooth_flitering},
                                {"edge-extraction",&edge_extraction},
                                {"sharpening-filtering",&sharpen_flitering},
                                {"bilinear-interpolation",&bilinear_interpolation},
                                {"scale-Part-Average",&scale_part_average },
                                {"relief",&relief},
                                {"sketch",&sketch}, 
								{"frozen",&frozen},
                              };

static void usage(char **argv)
{
    printf(
        "Usage: %s [Options]\n\n"
        "Options:\n"
        "-m, --mipi                   Mipi camera: 1 -- mipi1 cmaera; 2 -- mipi2 camera, default: 1"
        "-t, --type                   Mipi camera type: ov9750 or imx258, default: ov9750"
        "-w, --width                  Destination images's width\n"
        "-h, --height                 Destination images's height\n"
        "-r, --rotate                 Image rotation degree, the value should be 90, 180 or 270\n"
        "-s, --scene                  Scene: 0 -- histogram equalization; 1 -- median smooth filtering; 2 -- sharpening filtering; 4 -- bilinear_interpolation; default: 0\n"
        "-V, --vflip                  Vertical Mirror\n"
        "-H, --hflip                  Horizontal Mirror\n"
        "-c, --crop                   Crop, format: x,y,w,h\n"
        "\n",
        argv[0]);
}

static const char *short_options = "m:t:w:h:r:s:VHc";

static struct option long_options[] = {
    {"mipi", required_argument, NULL, 'm'},
    {"type", required_argument, NULL, 't'},
    {"width", required_argument, NULL, 'w'},
    {"height", required_argument, NULL, 'h'},
    {"rotate", required_argument, NULL, 'r'},
    {"scene", required_argument, NULL, 's'},
    {"vflip", no_argument, NULL, 'V'},
    {"hflip", no_argument, NULL, 'H'},
    {"crop", required_argument, NULL, 'c'},
    {NULL, 0, NULL, 0}
};

struct Box
{
	float x0;
	float y0;
	float x1;
	float y1;
	int class_idx;
	float score;
};

static void parse_crop_parameters(char *crop, __u32 *cropx, __u32 *cropy, __u32 *cropw, __u32 *croph)
{
    char *p, *buf;
    const char *delims = ".,";
    __u32 v[4] = {0,0,0,0};
    int i = 0;

    buf = strdup(crop);
    p = strtok(buf, delims);
    while(p != NULL) {
        v[i++] = atoi(p);
        p = strtok(NULL, delims);

        if(i >=4)
            break;
    }

    *cropx = v[0];
    *cropy = v[1];
    *cropw = v[2];
    *croph = v[3];
}

void get_input_data_ssd(const Mat image, float* input_data, int img_h, int img_w)
{
	cv::Mat img = image.clone();

	if (img.empty())
	{
		std::cerr << "Failed to read image file " <<  ".\n";
		return;
	}

	cv::resize(img, img, cv::Size(img_h, img_w));
	img.convertTo(img, CV_32FC3);
	float* img_data = (float*)img.data;
	int hw = img_h * img_w;

	float mean[3] = { 127.5, 127.5, 127.5 };
	for (int h = 0; h < img_h; h++)
	{
		for (int w = 0; w < img_w; w++)
		{
			for (int c = 0; c < 3; c++)
			{
				input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
				img_data++;
			}
		}
	}
}


void post_process_ssd(const Mat img, float threshold, float* outdata, int num, Mat & image_dst)
{
	const char* class_names[] = { "background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
								 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
								 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
								 "sofa",       "train",     "tvmonitor" };

	//cv::Mat img = cv::imread(image_file);
	image_dst = img.clone();
	int raw_h = img.size().height;
	int raw_w = img.size().width;
	std::vector<Box> boxes;
	int line_width = raw_w * 0.005;
	printf("detect result num: %d \n", num);
	for (int i = 0; i < num; i++)
	{
		if (outdata[1] >= threshold)
		{
			Box box;
			box.class_idx = outdata[0];
			box.score = outdata[1];
			box.x0 = outdata[2] * raw_w;
			box.y0 = outdata[3] * raw_h;
			box.x1 = outdata[4] * raw_w;
			box.y1 = outdata[5] * raw_h;
			boxes.push_back(box);
			printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
			printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
		}
		outdata += 6;
	}
	for (int i = 0; i < (int)boxes.size(); i++)
	{
		Box box = boxes[i];
		cv::rectangle(image_dst, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
			line_width);
		std::ostringstream score_str;
		score_str << box.score;
		std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::rectangle(image_dst,
			cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
				cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 0), CV_FILLED);
		cv::putText(image_dst, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	//    cv::imwrite(save_name, img);
	//    std::cout << "======================================\n";
	//    std::cout << "[DETECTED IMAGE SAVED]:\t" << save_name << "\n";
	//    std::cout << "======================================\n";
}

int main(int argc, char **argv)
{
    int ret = -1, c, r;
    char v4l2_dev[64], isp_dev[64];
    char index = -1;
    pthread_t id;
    Mat image;
    Mat image_dst;
    //void (*p_scene_processing)(const Mat src, Mat &dst);
    struct timeval t0, t1;

    /* Window -- create */
    fcv::namedWindow(wintitle);
    fcv::moveWindow(wintitle, 720, 480);

    /* MIPI Camera -- default values */
    int mipi = 1;    /* main camera */
    enum CAM_TYPE type = CAM_OV9750; /* HD camera sensor: OV9750 */
    int scene = 0;
    __u32 width = 640, height = 480; /* resolution: 640x480 */
    RgaRotate rotate = RGA_ROTATE_NONE; /* No rotation */
    __u32 cropx = 0, cropy = 0, cropw = 0, croph = 0;
    int vflip = 0, hflip = 0; /* no flip */

    while((c = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
        switch (c) {
        case 'm':
            mipi = atoi(optarg);
            break;
        case 't':
            if(strncmp(optarg, "ov9750", 6) == 0)
                type = CAM_OV9750;
            if(strncmp(optarg, "imx258", 6) == 0)
                type = CAM_IMX258;
            break;
        case 'w':
            width = atoi(optarg);
            break;
        case 'h':
            height = atoi(optarg);
            break;
        case 'r':
            r = atoi(optarg);
            switch(r) {
            case 0:
                rotate = RGA_ROTATE_NONE;
                break;
            case 90:
                rotate = RGA_ROTATE_90;
                break;
            case 180:
                rotate = RGA_ROTATE_180;
                break;
            case 270:
                rotate = RGA_ROTATE_270;
                break;
            default:
                printf("roate %d is not supported\n", r);
                return -1;
            }
            break;
        case 's':
            scene = atoi(optarg);
            break;
        case 'V':
            vflip = 1;
            break;
        case 'H':
            hflip = 1;
            break;
        case 'c':
            parse_crop_parameters(optarg, &cropx, &cropy, &cropw, &croph);
            break;
        default:
            usage(argv);
            return 0;
        }
    }

    if (type == CAM_IMX258)
    {
        type = CAM_OV9750;
        printf ("IMX258 is not supported currently. Use OV9750 instead!\n");
    }

    if (scene < HISTOGRAM_EQUALIZATION || scene >= SCENE_MAX)
    {
        scene = HISTOGRAM_EQUALIZATION;
        printf ("Scene is not supported currently. Use HISTOGRAM_EQUALIZATION instead!\n");
    }

    fcv::namedWindow("mssd", WINDOW_AUTOSIZE);
    //p_scene_processing = scene_processing_arr[scene].scene_processing_fun;
    /* V4L2 device */
    sprintf(v4l2_dev, "/dev/video%d", 4 * (mipi - 1) + 2);
    sprintf(isp_dev, "/dev/video%d", 4 * (mipi - 1) + 1);

    printf("width = %u, height = %u, rotate = %u, vflip = %d, hflip = %d, crop = [%u, %u, %u, %u]\n",
           width, height, rotate, vflip, hflip, cropx, cropy, cropw, croph);



    /* MIPI Camera -- initialization */
    v4l2Camera v4l2(width, height, rotate, vflip, hflip, cropx, cropy, cropw, croph, V4L2_PIX_FMT_NV12);
    image.create(cv::Size(RGA_ALIGN(width, 16), RGA_ALIGN(height, 16)), CV_8UC3);
    image_dst.create(cv::Size(RGA_ALIGN(width, 16), RGA_ALIGN(height, 16)), CV_8UC3);

    ret = v4l2.init(v4l2_dev, isp_dev, type);
    if(ret < 0)
    {
        printf("v4l2Camera initialization failed.\n");
        return ret;
    }

    /* MIPI Camera -- open stream */
    ret = v4l2.streamOn();
    if(ret < 0)
        return ret;


	const std::string root_path = "./";
	std::string model_file;
	const char* device = nullptr;

	int res;
	if (model_file.empty())
	{
		model_file = root_path + DEF_MODEL;
		std::cout << "model file not specified,using " << model_file << " by default\n";
	}

	if (init_tengine() < 0)
	{
		std::cout << " init tengine failed\n";
		return 1;
	}
	if (request_tengine_version("0.9") != 1)
	{
		std::cout << " request tengine version failed\n";
		return 1;
	}

	// create graph
	graph_t graph = create_graph(nullptr, "tengine", DEF_MODEL);

	if (graph == nullptr)
	{
		std::cout << "Create graph failed\n";
		std::cout << " ,errno: " << get_tengine_errno() << "\n";
		return 1;
	}

	if (device != nullptr)
	{
		set_graph_device(graph, device);
	}

	// dump_graph(graph);

		// input
	int img_h = 300;
	int img_w = 300;
	int img_size = img_h * img_w * 3;
	float* input_data = (float*)malloc(sizeof(float) * img_size);

	int node_idx = 0;
	int tensor_idx = 0;
	tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
	tensor_t out_tensor;
	if (input_tensor == nullptr)
	{
		std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
		return -1;
	}

	int dims[] = { 1, 3, img_h, img_w };
	set_tensor_shape(input_tensor, dims, 4);
	ret = prerun_graph(graph);
	if (ret != 0)
	{
		std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
		return 1;
	}

	int repeat_count = 1;
	const char* repeat = std::getenv("REPEAT_COUNT");
	//std::getenv(const char*name)搜索name所指向的环境字符串并返回相关的值给字符串
	//unsigned long int strtoul(const char* str, char** endptr, int base) 把参数 str 所指向的字符串根据给定的 base 转换为一个无符号长整数（类型为 unsigned long int 型），base 必须介于 2 和 36（包含）之间，或者是特殊值 0
	if (repeat)
		repeat_count = std::strtoul(repeat, NULL, 10);


    while(1) {
        gettimeofday(&t0, NULL);
        /* MIPI Camera -- read video frame */
        if(ret = v4l2.readFrame(V4L2_PIX_FMT_RGB24,image) < 0)
            return ret;


		// warm up
		get_input_data_ssd(image, input_data, img_h, img_w);
		set_tensor_buffer(input_tensor, input_data, img_size * 4);
		ret = run_graph(graph, 1);
		if (ret != 0)
		{
			std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
			return 1;
		}

		struct timeval t0, t1;
		float total_time = 0.f;
		for (int i = 0; i < repeat_count; i++)
		{
			get_input_data_ssd(image, input_data, img_h, img_w);

			gettimeofday(&t0, NULL);
			run_graph(graph, 1);
			gettimeofday(&t1, NULL);
			float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
			total_time += mytime;
		}
		std::cout << "--------------------------------------\n";
		std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";
		out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
		int out_dim[4];
		ret = get_tensor_shape(out_tensor, out_dim, 4);
		if (ret <= 0)
		{
			std::cout << "get tensor shape failed, errno: " << get_tengine_errno() << "\n";
			return 1;
		}
		float* outdata = (float*)get_tensor_buffer(out_tensor);
		int num = out_dim[1];
		float show_threshold = 0.5;

		post_process_ssd(image, show_threshold, outdata, num, image_dst);


		fcv::imshow(wintitle, image, NULL);

        fcv::imshow("mssd", image_dst, NULL);

        //fcv::imshow(wintitle_noise, image_noise, NULL);
        gettimeofday(&t1, NULL);
        fcv::waitKey(1);
        long elapse = ((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;

    }

    image.release();
    image_dst.release();

    /* MIPI Camera -- close stream */
    ret = v4l2.streamOff();


    if(ret < 0)
        return ret;

	release_graph_tensor(out_tensor);
	release_graph_tensor(input_tensor);

	ret = postrun_graph(graph);
	if (ret != 0)
	{
		std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
		return 1;
	}
	free(input_data);
	destroy_graph(graph);
	release_tengine();

    return 0;
}



