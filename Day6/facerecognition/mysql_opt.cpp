#include <string>
#include "mtcnn.hpp"
#include "lightcnn.hpp"
#include "mtcnn_utils.hpp"
#include "mysql_opt.hpp"


/* get face image with face detection results */
void get_face_image(fcv::Mat& img,std::vector<face_box>& face_list, fcv::Mat& faceImg,fcv::Point* leftTopP,fcv::Point* rightBottomP)
{
    float boxSize=0;
    float maxWidth=0;
    int maxI = 0;
    float faceWidth = 0;
    fcv::Size faceSize(FACEWIDTH,FACEHEIGHT);


    /* Select face with largest size */
    for ( unsigned int i = 0; i < face_list.size(); i++)
    {
        face_box &box = face_list[i];
        boxSize = box.x1 - box.x0;
        if (boxSize>maxWidth)
        {
            maxWidth = boxSize;
            maxI = i;
        }
    }
    face_box &box = face_list[maxI];

    /* calculate face width in oringal image */
    faceWidth = (box.landmark.x[1] - box.landmark.x[0])/0.53194925;

    /* calculate face roi for image cropping */
    fcv::Rect roi;
    roi.x = box.landmark.x[0]-(0.224152*faceWidth);
    if (roi.x < 0)
    {
        roi.x = 0;
    }

    if (roi.x + faceWidth> img.cols)
    {
        roi.x = img.cols - faceWidth;
    }
    roi.y = (box.landmark.y[0]+box.landmark.y[1])/2-(0.2119465*faceWidth);

    if (roi.y < 0)
    {
        roi.y = 0;
    }
    if (roi.y + faceWidth> img.rows)
    {
        roi.y = img.rows - faceWidth;
    }

    roi.width = faceWidth;
    roi.height = faceWidth;
    std::cout << "FaceWidth=" << faceWidth << endl;

    if (leftTopP)
    {
        leftTopP->x = roi.x;
        leftTopP->y = roi.y;
    }

    if (rightBottomP)
    {
        rightBottomP->x = roi.x+roi.width;
        rightBottomP->y = roi.y+roi.height;
    }

    /* crop face image */
    fcv::Mat cropA = img(roi);
    /* resize image for lcnn input */
    fcv::resize(cropA,faceImg,faceSize);

}

struct face get_face_from_db_by_id(MYSQL *mysql,int id)
{
    char sql[2048] = {0};
    MYSQL_RES *result;
    MYSQL_ROW row;
    struct face face_tmp;

    memset((char*)&face_tmp,0,sizeof(face_tmp));
    sprintf(sql,"SELECT  photo_path,name, feature FROM face_tbl WHERE face_id=%d",id);

    if(mysql_query(mysql, sql) != 0)
    {
        printf( "Failed to query face id(%d): %d\n",id);
        return face_tmp;
    }

    result = mysql_store_result(mysql);
    if(result == NULL)
    {
        printf( "Failed to query face id(%d): %d\n",id);
        return face_tmp;
    }
      
    if((row = mysql_fetch_row(result)) != NULL)
    {
        face_tmp.feature= (float *)malloc(sizeof(float)*FEATURESIZE);
        strncpy(face_tmp.photo_path,row[0],PHOTE_LENTH-1);
        strncpy(face_tmp.name,row[1],NAME_LENTH-1);
        memcpy(face_tmp.feature, row[2],sizeof(float)*FEATURESIZE);
     }
    
     mysql_free_result(result);
     return face_tmp;

}

