#include "visualization.h"

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

    }
}