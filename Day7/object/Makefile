export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig:$PKG_CONFIG_PATH

LDDFLAGS= -lrockchip_rga -lrockchip_mpp -ldrm -lcam_engine_cifisp -lcam_ia -lrockchip_vendor `pkg-config --libs fastcv` `pkg-config --libs gtk+-3.0` `pkg-config --libs tengine`
INCFLAGS= `pkg-config --cflags fastcv` `pkg-config --cflags gtk+-3.0` `pkg-config --cflags tengine`

all:
	g++ -g -c mssd.cpp  ${INCFLAGS}
	g++ mssd.o  ${LDDFLAGS} -o test_mssd
