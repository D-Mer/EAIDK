#include "fastcv.hpp"
#include <stdio.h>
#include "openai_io.hpp"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cstring> 

using namespace fcv;
using namespace std;

#define DEFAULT_IMAGE_FILE1 "0.jpg"
#define DEFAULT_IMAGE_FILE2 "50.jpg"


int main(int argc, char* argv[]) {
	Mat image1 = fcv::imread(DEFAULT_IMAGE_FILE1);
	for (int i = 0; i < 100; i++) {
		fcv::imwrite(to_string(i)+".jpg", image1);
	}

}