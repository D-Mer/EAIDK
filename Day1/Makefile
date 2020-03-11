export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig:$PKG_CONFIG_PATH

LDDFLAGS=`pkg-config --libs fastcv`
INCFLAGS= `pkg-config --cflags fastcv` `pkg-config --cflags gtk+-3.0`
all:
	g++ -c test.cpp ${INCFLAGS}
	g++ test.o ${LDDFLAGS} -o test
