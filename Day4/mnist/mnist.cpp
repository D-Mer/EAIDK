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
 * Author: chunyinglv@openailab.com
 */
#include <unistd.h>

#include <iostream> 
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>


#include "tengine_c_api.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "common_util.hpp"

#define PRINT_TOP_NUM            1

const float channel_mean[3]={127.5,127.5,127.5};

//using namespace TEngine;

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}
void get_input_data(const char *image_file, float *input_data,
        int img_h, int img_w, const float* mean, float scale)
{
    cv::Mat sample = cv::imread(image_file, -1);
    if (sample.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
    cv::Mat img;
    if (sample.channels() == 4) 
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img=sample;
    }
    
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c])*scale;
                img_data++;
            }
        }
    }
}

static inline std::vector<int> Argmax1(const std::vector<float> &v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}
   
int main(int argc, char * argv[])
{
    if (argc < 3)
    {
        std::cout << "[Usage]: " << argv[0] << " <proto> <caffemodel> <img>\n";
        return 0;
    }

    // init tengine
    init_tengine();
    if (request_tengine_version("0.9") < 0)
        return 1;

    // load model
    const char *model_name = "mnist";
    std::string mdl_name_ = argv[1];
    std::string image_file = argv[2];
	graph_t graph = create_graph(nullptr, "tengine", mdl_name_.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph failed.\n";
        return 1;
    }
   int img_h=28;
   int img_w=28;
   float  * input_data=(float*) malloc (sizeof(float) * img_h *img_w *3);
   tensor_t input_tensor = get_graph_input_tensor(graph, 0,0);
   if(input_tensor == nullptr)
    {
        std::cout << "Get input tensor failed\n";
        return 1;
    }
   get_input_data( image_file.c_str(), input_data, img_h, img_w, channel_mean,1.f/255);
   const char *input_tensor_name = "data";
  
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    prerun_graph(graph);
    set_tensor_buffer(input_tensor, input_data, 3*28*28 * 4);

    run_graph(graph, 1);
    // dump_graph(graph);
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float *data = (float *)get_tensor_buffer(output_tensor);
    float *end = data + 10;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax1(result, PRINT_TOP_NUM);

    for (unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

	    std::cout<<"Predict Result:" <<idx << "\n";
    }
    free(input_data);
    postrun_graph(graph);  
   destroy_runtime_graph(graph);
   remove_model(model_name);
   release_tengine_library();
   return 0;
}
