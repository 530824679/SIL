#include "visualization.h"
#include <iostream>

namespace perception
{
    void Visualizer::visualize2D(cv::Mat &image, std::vector<BoxInfo> &boxes, const std::vector<std::string> &classNames)
    {
        for (const BoxInfo& box : boxes)
        {
            cv::rectangle(image, box.bbox, cv::Scalar(229, 160, 21), 2);

            int x = box.bbox.x;
            int y = box.bbox.y;

            int conf = (int)(box.score * 100);
            int classId = box.classId;
            std::string label = classNames[classId] + " 0." + std::to_string(conf);

            int baseline = 0;
            cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
            cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);

            cv::putText(image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
        }
    }

    void Visualizer::visualize3D(cv::Mat &image, std::vector<KeypointsInfo> &detections)
    {
        for (const KeypointsInfo& keypoint : detections)
        {

            std::cout << "123456"<< std::endl;
            int pt5_x = keypoint.pt2.x;
            int pt5_y = keypoint.pt1.y;
            int pt6_x = keypoint.pt1.x;
            int pt6_y = keypoint.pt2.y;
            std::cout << keypoint.pt1.x<< std::endl;
            std::cout << keypoint.pt1.y<< std::endl;
            std::cout << keypoint.pt2.x<< std::endl;
            std::cout << keypoint.pt2.y<< std::endl;
            std::cout << keypoint.pt3.x<< std::endl;
            std::cout << keypoint.pt3.y<< std::endl;
            std::cout << keypoint.pt4.x<< std::endl;
            std::cout << keypoint.pt4.y<< std::endl;
            cv::line(image,cv::Point(keypoint.pt1.x,keypoint.pt1.y),cv::Point(pt5_x,pt5_y),cv::Scalar(0, 255, 0),1);
            cv::line(image,cv::Point(keypoint.pt1.x,keypoint.pt1.y),cv::Point(pt6_x,pt6_y),cv::Scalar(0, 255, 0),1);
            cv::line(image,cv::Point(keypoint.pt2.x,keypoint.pt2.y),cv::Point(pt5_x,pt5_y),cv::Scalar(0, 255, 0),1);
            cv::line(image,cv::Point(keypoint.pt2.x,keypoint.pt2.y),cv::Point(pt6_x,pt6_y),cv::Scalar(0, 255, 0),1);
            cv::line(image,cv::Point(keypoint.pt4.x,keypoint.pt4.y),cv::Point(pt5_x,pt5_y),cv::Scalar(0, 255, 0),1);
            cv::line(image,cv::Point(keypoint.pt3.x,keypoint.pt3.y),cv::Point(keypoint.pt4.x,keypoint.pt4.y),cv::Scalar(0, 255, 0),1);   
            cv::line(image,cv::Point(keypoint.pt3.x,keypoint.pt3.y),cv::Point(keypoint.pt3.x,keypoint.pt2.y),cv::Scalar(0, 255, 0),1);                      
        }


    }
}