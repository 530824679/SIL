#include <stdlib.h>
#include "../include/runable.hpp"



int main()
{
    perception::odet_sil od;
    od.init();
    od.process("/home/qzx/code/SIL_rebuild/test/test_front.png");


}