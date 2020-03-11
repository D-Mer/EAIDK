export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig:$PKG_CONFIG_PATH

LDDFLAGS= -lrockchip_rga -lrockchip_mpp -ldrm -lcam_engine_cifisp -lcam_ia -lrockchip_vendor `pkg-config --libs fastcv` `pkg-config --libs gtk+-3.0` `pkg-config --libs tengine` -lrtspclient -lpthread
INCFLAGS= `pkg-config --cflags fastcv` `pkg-config --cflags gtk+-3.0` `pkg-config --cflags tengine`

all:
	g++ -g -c mssd.cpp  ${INCFLAGS}
	g++ mssd.o  ${LDDFLAGS} -o test_mssd
	g++ -c mipi_cam.cpp image_handle.cpp mipi-demo.cpp ${INCFLAGS}
	g++ mipi_cam.o image_handle.o mipi-demo.o ${LDDFLAGS} -o mipi-demo