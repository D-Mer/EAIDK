#include <string>
#include "mtcnn.hpp"
#include "lightcnn.hpp"
#include "mtcnn_utils.hpp"
#include "mysql_opt.hpp"

#define FACE_CONFIG_PATH "./face/db_config"

int main(int argc, char **argv)
{
     
    /* Variable definition */    
    std::string model_dir = "./models/";
    int ret;
    MYSQL *mysql;         
    FILE *fp;

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
        printf("mariadb init fail!\n");
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
    mysql_query(mysql,"drop database FACE");
    if (0 != mysql_query(mysql,"create database if not exists FACE"))
    {
        printf("create db FACE fail!\n");
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

    if (0 != mysql_query(mysql,"CREATE TABLE face_tbl( "
                                "face_id INT NOT NULL AUTO_INCREMENT, "
                                "photo_path VARCHAR(100) NOT NULL, "
                                "name VARCHAR(100) NOT NULL, "
                                "feature BLOB NOT NULL, "
                                "PRIMARY KEY ( face_id ))ENGINE=InnoDB DEFAULT CHARSET=utf8; "))
    {
        printf("create db_tbl face_tbl fail!\n");
        release_tengine_library();
        mysql_close(mysql);
        return -1;    
    }
      
    if((fp = fopen(FACE_CONFIG_PATH,"r")) == NULL) 
    {
        printf("open db_config file fail!\n");
        release_tengine_library();
        mysql_close(mysql);
        return -1;    
    } 
  
    while (!feof(fp)) 
    {
        char StrLine[1024];
        char *p = NULL;
        fcv::Mat photo;
        char photo_name[NAME_LENTH];
        char photo_path_tmp[PHOTE_LENTH-7];
        char photo_path[PHOTE_LENTH] = "./face/";
        char sql[2048] = {0};
        char *end;
        std::vector<face_box> photo_info;
        fcv::Mat faceImage;
        float *feature;
        
        fgets(StrLine,1024,fp);

        for (int i = strlen(StrLine)-1; i>=0; i--)
        {
            if (StrLine[i] != '\r' && StrLine[i] != '\n')
            {
                break;
            }
            StrLine[i] = 0;
        }
        p = strchr(StrLine,':');
        if ( p == NULL )
        {
            continue;
        }
        *p++ = 0;

        strcpy(photo_name,StrLine);
        strcpy(photo_path_tmp,p);
        strcat(photo_path,photo_path_tmp);

        photo = fcv::imread(photo_path);
        if (photo.empty())
        {
            continue;
        }
        detector.detect(photo, photo_info);
        
        if (photo_info.size()==0)
        {
            continue;
        }
        /* Get face from original image */
        get_face_image(photo, photo_info, faceImage,NULL,NULL); 
  
        feature = (float *)malloc(sizeof(float)*FEATURESIZE);

        faceFeature.featureExtract(faceImage,feature);

        sprintf(sql, "insert into face_tbl(photo_path,name,feature) values('%s','%s',",photo_path,photo_name); 
        end = sql; 
        end += strlen(sql);

        *end++ = '\'';

        end += mysql_real_escape_string(mysql, end, (char*)feature, sizeof(float)*FEATURESIZE); 
        *end++ = '\''; 
        *end++ = ')';  
        if (0 != mysql_real_query(mysql,sql,(unsigned int)(end-sql)))
        {
            printf("insert fail!\n");
            free(feature);
            feature = NULL;
            continue;   
        }
        free(feature);
        feature = NULL;
    } 

    printf("mysql init successfully!\n");
    fclose(fp);
    /* Tengine -- deinitialization */
    mysql_close(mysql);
    release_tengine_library();
    return 0;
}

