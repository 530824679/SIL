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

    void visualize3D(cv::Mat &image, std::vector<KeypointsInfo> &detections)
    {
        for (const auto &item : detections)
        {
            float ang = item.bbox.ang;
            float cx = (item.bbox.x1 + item.bbox.x2) / 2;
            float cy = (item.bbox.y1 + item.bbox.y2) / 2;
            float height = (item.bbox.x2 - item.bbox.x1);
            float width = (item.bbox.y2 - item.bbox.y1);
            float anglePi = ang / 180 * PI;
            anglePi = anglePi < PI ? anglePi : anglePi - PI;
            float cosA = cos(anglePi);
            float sinA = sin(anglePi);
            float x1 = cx - 0.5 * width;
            float y1 = cy - 0.5 * height;

            float x0 = cx + 0.5 * width;
            float y0 = y1;

            float x2 = x1;
            float y2 = cy + 0.5 * height;

            float x3 = x0;
            float y3 = y2;

            int x0n = floor((x0 - cx) * cosA - (y0 - cy) * sinA + cx);
            int y0n = floor((x0 - cx) * sinA + (y0 - cy) * cosA + cy);

            int x1n = floor((x1 - cx) * cosA - (y1 - cy) * sinA + cx);
            int y1n = floor((x1 - cx) * sinA + (y1 - cy) * cosA + cy);

            int x2n = floor((x2 - cx) * cosA - (y2 - cy) * sinA + cx);
            int y2n = floor((x2 - cx) * sinA + (y2 - cy) * cosA + cy);

            int x3n = floor((x3 - cx) * cosA - (y3 - cy) * sinA + cx);
            int y3n = floor((x3 - cx) * sinA + (y3 - cy) * cosA + cy);

            cv::line(image, cv::Point(x0n, y0n), cv::Point(x1n, y1n), cv::Scalar(0, 0, 255), 3, 8, 0);
            cv::line(image, cv::Point(x1n, y1n), cv::Point(x2n, y2n), cv::Scalar(255, 0, 0), 3, 8, 0);
            cv::line(image, cv::Point(x2n, y2n), cv::Point(x3n, y3n), cv::Scalar(0, 0, 255), 3, 8, 0);
            cv::line(image, cv::Point(x3n, y3n), cv::Point(x0n, y0n), cv::Scalar(255, 0, 0), 3, 8, 0);
        }
    }
}