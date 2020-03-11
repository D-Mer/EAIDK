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
 * Author: minglu@openailab.comm
 *
 */
#include <string>
#include <sys/time.h>
#include "mtcnn.hpp"
#include "lightcnn.hpp"
#include "mtcnn_utils.hpp"
#include "mipi_cam.hpp"
#include "mysql_opt.hpp"

#define ONE_TO_N 0
#define ONE_TO_ONE 1
#define ONE_TO_ONE_POLLING 2

const string wintitle = "mipi-camera";
//const string ipcUrl="rtsp://192.168.88.76/cam/realmonitor?channel=1&subtype=0"
//const string ipcUrl="rtsp://192.168.88.174/video1";
//const string ipcUser="admin";
//const string ipcPassword="12345678";

//const string ipcUrl="rtsp://192.168.88.101/cam/realmonitor?channel=1&subtype=0";
//const string ipcUser="admin";
//const string ipcPassword="oalb2018";

static void usage(char **argv)
{
    printf(
        "Usage: %s [Options]\n\n"
        "Options:\n"
        "-m, --mipi                   Mipi camera: 1 -- mipi1 cmaera; 2 -- mipi2 camera, default: 1"
        "-p, --photo                  Photo\n"
        "\n",
        argv[0]);
}

static const char *short_options = "m:p:";

static struct option long_options[] = {
    {"mipi", required_argument, NULL, 'm'},
    {"photo", required_argument, NULL, 'r'},
    {NULL, 0, NULL, 0}
};

/* calculate cosine distance of two vectors */
float cosine_dist(float* vectorA, float* vectorB, int size)
{
    float Numerator=0;
    float Denominator1=0;
    float Denominator2=0;
    float Similarity;
    for (int i = 0 ; i < size ; i++)
    {
        Numerator += (vectorA[i] * vectorB[i]);
        Denominator1 += (vectorA[i] * vectorA[i]);
        Denominator2 += (vectorB[i] * vectorB[i]);
    }

    Similarity = Numerator/sqrt(Denominator1)/sqrt(Denominator2);

    return Similarity;
}

int main(int argc, char **argv)
{        
    /* Variable definition */    
    std::string model_dir = "./models/";

    int ret,c;
    int index = 0;
    char v4l2_dev[64], isp_dev[64];

    fcv::Mat image;
    fcv::Mat face_show;
    int primary_key = -1;
    int mode = ONE_TO_N;
    struct face face_db[PHOTO_NUM_MAX];
    int photo_max = 0;
    fcv::Point mode_p;
    fcv::Point name_p;
    char mode_show[30];

    fcv::Size face_show_Size(FACEWIDTH,FACEWIDTH);
    /* Window -- create */
    fcv::namedWindow(wintitle);
    fcv::moveWindow(wintitle, 720, 480);

    /* MIPI Camera -- default values */
    int mipi = 1;    /* main camera */
    CAM_TYPE type = CAM_OV9750; /* HD camera sensor: OV9750 */
    __u32 width = 640, height = 480; /* resolution: 640x480 */
    RgaRotate rotate = RGA_ROTATE_NONE; /* No rotation */
    __u32 cropx = 0, cropy = 0, cropw = 0, croph = 0;
    int vflip = 0, hflip = 0; /* no flip */

    MYSQL *mysql;         
    FILE *fp;

    int b_detect = 0;
    fcv::Mat image_capture_ROI;
    fcv::Mat capture_mask;  
    fcv::Mat faceImage;
    struct timeval t_polling_start,t_polling_end,t_detect_start,t_detect_end;

    while((c = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
        switch (c) {
        case 'm':
            mipi = atoi(optarg);
            break;
        case 'p':
            mode = ONE_TO_ONE;
            primary_key = atoi(optarg);
            break;
        default:
            usage(argv);
            return 0;
        }
    }

    if (mode == ONE_TO_ONE&&(primary_key < 1 || primary_key > PHOTO_NUM_MAX))
    {
        if (primary_key == 0)
        {
            mode = ONE_TO_ONE_POLLING;
        }
        else
        {
            primary_key = 1;
            printf ("primary_key is invalid. Use default primary_key(1)!\n");

        }
    }
    image.create(cv::Size(RGA_ALIGN(width, 16), RGA_ALIGN(height, 16)), CV_8UC3);

    /* Tengine -- initialization */
    printf("Tengine version: %s\n", get_tengine_version());
    init_tengine_library();
    if(request_tengine_version("0.1")<0)
    {
        release_tengine_library();
        return -2;
    }
    
    /* MTCNN -- default value */
    int min_size=60;
    float conf_p=0.6;
    float conf_r=0.7;
    float conf_o=0.8;
    float nms_p=0.5;
    float nms_r=0.7;
    float nms_o=0.7;
    
    /* MTCNN -- initialization */
    mtcnn detector(min_size,conf_p,conf_r,conf_o,nms_p,nms_r,nms_o);
    /* MTCNN -- load models */
    ret = detector.load_3model(model_dir);
    if(ret != 0)
    {
        std::cout << "can not load mtcnn models." << endl;
        release_tengine_library();
        return -1;
    }
    
    /* LightCNN -- initialization */
    lightcnn faceFeature;
    ret = faceFeature.init(model_dir);
    if(ret <0)
    {
        std::cout << "can not initialize light cnn."<< endl;
        release_tengine_library();
        return -1;
    }

    /* Mariadb -- initialization */
    mysql= mysql_init(NULL);
    if (mysql == NULL)
    {
        release_tengine_library();
        return -1;
    }
    if (!mysql_real_connect(mysql, "localhost", "root", "123456","test", 0, "/var/lib/mysql/mysql.sock", 0))
    {
        printf("connnect mariadb fail!\n");
        release_tengine_library();
        mysql_close(mysql);
        return -1;
    }

    if (0 != mysql_query(mysql,"use FACE"))
    {
        printf("use db FACE fail!\n");
        release_tengine_library();
        mysql_close(mysql);
        return -1;    
    }

	//////////////////////////////////////// TODO ///////////////////////////////////
    if (mode == ONE_TO_ONE)
    {
		face_db[0] = get_face_from_db_by_id(mysql, primary_key);
		if (face_db[0].feature == NULL){
			printf("db FACE is empty!\n");
			release_tengine_library();
			mysql_close(mysql);
			return -1;
		}
		face_show = fcv::imread(face_db[0].photo_path);
		fcv::resize(face_show, face_show, face_show_Size);
		strcpy(mode_show, "1:1");
    }


    else if (mode == ONE_TO_ONE_POLLING)
    {
        int i = 0;
        for (i = 0; i< PHOTO_NUM_MAX; i++)
        {
            face_db[i] = get_face_from_db_by_id(mysql,i+1);
            if (face_db[i].feature == NULL)
            { 
                break;
            }
        }
        photo_max = i;
        if (photo_max == 0)
        {
            printf("db FACE is empty!\n");
            release_tengine_library();
            mysql_close(mysql);
            return -1;
        }
        face_show = fcv::imread(face_db[0].photo_path);
        fcv::resize(face_show,face_show,face_show_Size);
        strcpy(mode_show,"1:1");  
    }

    else if (mode == ONE_TO_N)
    {
        int i = 0;
        for (i = 0; i< PHOTO_NUM_MAX; i++)
        {
            face_db[i] = get_face_from_db_by_id(mysql,i+1);
            if (face_db[i].feature == NULL)
            {
                break;
            }
        }
        photo_max = i;
        if (photo_max == 0)
        {
            printf("db FACE is empty!\n");
            release_tengine_library();
            mysql_close(mysql);
            return -1;
        }
        strcpy(mode_show,"1:N");
    }
    mode_p.x = image.cols/2-10;
    mode_p.y = 25;

    name_p.x = 0;
    name_p.y = image.rows-25;
    /* V4L2 device */
    sprintf(v4l2_dev, "/dev/video%d", 4 * (mipi - 1) + 2);
    sprintf(isp_dev, "/dev/video%d", 4 * (mipi - 1) + 1);

    printf("width = %u, height = %u, rotate = %u, vflip = %d, hflip = %d, crop = [%u, %u, %u, %u]\n",
           width, height, rotate, vflip, hflip, cropx, cropy, cropw, croph);



    /* MIPI Camera -- initialization */
    v4l2Camera v4l2(width, height, rotate, vflip, hflip, cropx, cropy, cropw, croph, V4L2_PIX_FMT_NV12);

    ret = v4l2.init(v4l2_dev, isp_dev, type);
    if(ret < 0)
    {
        printf("v4l2Camera initialization failed.\n");
        release_tengine_library();
        mysql_close(mysql);
        return ret;
    }

    /* MIPI Camera -- open stream */
    ret = v4l2.streamOn();
    if(ret < 0)
    {
        release_tengine_library();
        mysql_close(mysql);
        return ret;
    }

    gettimeofday(&t_polling_start, NULL);
    while(1) {
        std::vector<face_box> face_info;
        float *feature;
        fcv::Point leftTopP;
        fcv::Point rightBottomP;
        float similarity;
        float similarity_max;
        int index_max;
        //char* name_dst;
        fcv::Scalar name_color = {255,255,255,0};

        if(ret = v4l2.readFrame(V4L2_PIX_FMT_RGB24,image) < 0)
            continue;
       
        if (image.empty())
        {
            std::cerr<<"image is empty\n";
            continue;
        }

        //Faces need to be displayed for 1 second when be detected 
        if (b_detect)
        {
            gettimeofday(&t_detect_end, NULL);
           
            if (((t_detect_end.tv_sec * 1000000 + t_detect_end.tv_usec) - (t_detect_start.tv_sec * 1000000 + t_detect_start.tv_usec)) / 1000 < 1*1000)
            {
                image_capture_ROI = image(cv::Rect(0, FACEWIDTH, faceImage.cols, faceImage.rows));
                capture_mask = fcv::Mat(image_capture_ROI.rows,image_capture_ROI.cols,image_capture_ROI.depth(),Scalar(1));  
                faceImage.copyTo(image_capture_ROI, capture_mask);
                if (mode == ONE_TO_N)
                {
                    fcv::Mat imageROI = image(cv::Rect(0, 0, face_show.cols, face_show.rows));
                    fcv::Mat mask(imageROI.rows,imageROI.cols,imageROI.depth(),Scalar(1));  
                    face_show.copyTo(imageROI, mask);
                }
                goto showimage;
            }
        }
        b_detect = 0;

        if (mode == ONE_TO_ONE_POLLING)
        {
            gettimeofday(&t_polling_end, NULL);
        
            if (((t_polling_end.tv_sec * 1000000 + t_polling_end.tv_usec) - (t_polling_start.tv_sec * 1000000 + t_polling_start.tv_usec)) / 1000 > 10*1000)
            {
                index++;
                if (index >= photo_max)
                {
                    index = 0;
                }
                t_polling_start = t_polling_end;
                face_show = fcv::imread(face_db[index].photo_path);
                if (face_show.empty())
                {
                    printf("read photo[%s] fail!\n",face_db[index].photo_path);
                    mysql_close(mysql);
                    release_tengine_library();
                    image.release();
                    return -1;
                }
                fcv::resize(face_show,face_show,face_show_Size);
            }
        }
        /* MTCNN -- detect faces in image */
        detector.detect(image, face_info);
        if (face_info.size()==0)
        {
            //std::cout <<  "Can not detect face "<< endl;
            goto showimage;
        }
        
        /* Get face from original image */
        get_face_image(image, face_info, faceImage,&leftTopP,&rightBottomP); 

        /* extract feature from face image with light cnn*/
        feature = (float *)malloc(sizeof(float)*FEATURESIZE);
        faceFeature.featureExtract(faceImage,feature);

		////////////////////////////////// TODO //////////////////////////
        /* calculte similarity of two face features */
        if (mode == ONE_TO_ONE || mode == ONE_TO_ONE_POLLING){
			similarity_max = cosine_dist(feature, face_db[0].feature, FEATURESIZE);
        }
        
        else if (mode == ONE_TO_N)
        {
            similarity_max = cosine_dist(feature, face_db[0].feature, FEATURESIZE);
            index_max = 0;
            //name_dst = face_db[0].name;

            for (int i = 1; i<photo_max;i++)
            {
                similarity = cosine_dist(feature, face_db[i].feature, FEATURESIZE);
                if (similarity > similarity_max)
                {
                    similarity_max = similarity;
                    index_max = i;
                    //name_dst = face_db[i].name;
                }
            }
            
        }
        std::cout << "Similarity: " << similarity_max << endl;
        
        if (similarity_max > 0.55)
        {   
            name_color = {0,0,255,0};
            if(mode == ONE_TO_N)
            {
                face_show = fcv::imread(face_db[index_max].photo_path);
                if (face_show.empty())
                {
                     printf("read photo[%s] fail!\n",face_db[index].photo_path);
                     mysql_close(mysql);
                     release_tengine_library();
                     image.release();
                     return -1; 
                }

                fcv::resize(face_show,face_show,face_show_Size);

                fcv::Mat imageROI = image(cv::Rect(0, 0, face_show.cols, face_show.rows));
                fcv::Mat mask(imageROI.rows,imageROI.cols,imageROI.depth(),Scalar(1));  
                face_show.copyTo(imageROI, mask);
            }
            image_capture_ROI = image(cv::Rect(0, FACEWIDTH, faceImage.cols, faceImage.rows));
            capture_mask = fcv::Mat(image_capture_ROI.rows,image_capture_ROI.cols,image_capture_ROI.depth(),Scalar(1));  
            faceImage.copyTo(image_capture_ROI, capture_mask);
            b_detect = 1;
            gettimeofday(&t_detect_start, NULL);
        }
        fcv::rectangle(image, leftTopP, rightBottomP, name_color, 1, 4, 0);
        free(feature);
	showimage:

		////////////////////// TODO ////////////////////////////////////////
        if (mode == ONE_TO_ONE || mode == ONE_TO_ONE_POLLING)
        {
			fcv::resize(face_show, face_show, face_show_Size);

			fcv::Mat imageROI = image(cv::Rect(0, 0, face_show.cols, face_show.rows));
			fcv::Mat mask(imageROI.rows, imageROI.cols, imageROI.depth(), Scalar(1));
			face_show.copyTo(imageROI, mask);
            //fcv::putText(image, name_dst, name_p, cv::FONT_HERSHEY_COMPLEX, 1, name_color, 1, 8, 0);
        }

        fcv::putText(image, mode_show, mode_p, cv::FONT_HERSHEY_COMPLEX, 1, fcv::Scalar(0, 0, 255), 1, 8, 0);
        imshow(wintitle, image, NULL);
        waitKey(1);
        usleep(20000);
    }

    for(int i = 0;i<PHOTO_NUM_MAX;i++)
    {
        if (face_db[i].feature)
        {
            free(face_db[i].feature);
        }
    }
    /* Tengine -- deinitialization */
    mysql_close(mysql);
    release_tengine_library();
    image.release();
    return 0;
}

