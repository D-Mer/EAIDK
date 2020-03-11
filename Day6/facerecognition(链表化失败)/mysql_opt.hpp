#ifndef __MYSQL_OPT_HPP__
#define __MYSQL_OPT_HPP__

#include <mysql/mysql.h>

#define NAME_LENTH 100
#define PHOTE_LENTH 100
#define PHOTO_NUM_MAX 50


struct face{
    int face_id;
    char photo_path[PHOTE_LENTH];
    char name[NAME_LENTH];
    float* feature;
	struct face * next;
};

void get_face_image(fcv::Mat& img,std::vector<face_box>& face_list, fcv::Mat& faceImg,fcv::Point* leftTopP,fcv::Point* rightBottomP);
struct face get_face_from_db_by_id(MYSQL *mysql,int id);

struct face* get(face* head, int i);
struct face* insert(face* head, face* node);
#endif