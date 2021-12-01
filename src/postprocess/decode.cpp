#include "decode.h"
#include "visualization.h"
#include <algorithm>
using namespace perception;
Visualizer visualizer;
namespace perception
{
    double Decode::sigmoid(double x)
    {
        return (1 / (1 + exp(-x)));
    }

    std::vector<int> Decode::get_anchors(int grid_size)
    {
        std::vector<int> anchors(6);
        int anchor_80[6] = {10,13, 16,30, 33,23};
        int anchor_40[6] = {30,61, 62,45, 59,119};
        int anchor_20[6] = {116,90, 156,198, 373,326};

        if(grid_size == 80){
            anchors.insert(anchors.begin(), anchor_80, anchor_80 + 6);
        }
        else if(grid_size == 40){
            anchors.insert(anchors.begin(), anchor_40, anchor_40 + 6);
        }
        else if(grid_size == 20){
            anchors.insert(anchors.begin(), anchor_20, anchor_20 + 6);
        }
        return anchors;
    }
}
