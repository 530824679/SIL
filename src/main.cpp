#include <stdlib.h>
#include "../include/runable.hpp"



int main()
{
    perception::odet_sil od;
    od.init();
    od.process("/home/chenwei/HDD/Project/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/my_test/image_od/image_od/image/0000000003.bmp");

}