export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig:$PKG_CONFIG_PATH

LDDFLAGS=-lrockchip_vendor -lrockchip_rga -lrockchip_mpp -ldrm -lcam_engine_cifisp -lcam_ia -lpthread -lmysqlclient `pkg-config --libs fastcv` `pkg-config --libs gtk+-3.0` `pkg-config --libs tengine`
INCFLAGS= `pkg-config --cflags fastcv` `pkg-config --cflags gtk+-3.0` `pkg-config --cflags tengine`

all:
	g++ -c lightcnn.cpp mtcnn.cpp mtcnn_utils.cpp mipi_cam.cpp mysql_opt.cpp face_recognition.cpp mysql_init.cpp ${INCFLAGS}
	g++ lightcnn.o mtcnn.o mtcnn_utils.o mipi_cam.o mysql_opt.o face_recognition.o ${LDDFLAGS} -o face_recognition
	g++ lightcnn.o mtcnn.o mtcnn_utils.o mipi_cam.o mysql_opt.o mysql_init.o ${LDDFLAGS} -o mysql_init