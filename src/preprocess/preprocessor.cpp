#include "preprocessor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>
using namespace cv;

namespace perception
{
    size_t PreProcess::vectorProduct(const std::vector<int64_t>& vector)
    {
        if (vector.empty())
            return 0;

        size_t product = 1;
        for (const auto& element : vector)
            product *= element;

        return product;
    }


    void PreProcess::preprocessing2D(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
    {
        cv::Mat resizedImage, floatImage;
        cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
        letterbox(image, resizedImage, this->inputImageShape, cv::Scalar(114, 114, 114), this->isDynamicInputShape, false, true, 32);

        inputTensorShape[2] = resizedImage.rows;
        inputTensorShape[3] = resizedImage.cols;

        resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
        blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
        cv::Size floatImageSize {floatImage.cols, floatImage.rows};
        

        // hwc -> chw
        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(floatImage, chw);
    }

    void PreProcess::preprocessing3D(cv::Mat &image, BoxInfo box, float* output, std::vector<int> inputTensorShape)
    {
        //抠图box
        // cv::Mat image_crop = image(cv::Rect(box.bbox)); // 裁剪后的图
        int input_w = inputTensorShape[0];
        int input_h = inputTensorShape[1];
        cv::Mat image_crop = image(cv::Rect(int (max(box.bbox.x-16,0)),int (max(box.bbox.y-16,0)),box.bbox.width,box.bbox.height)); // 裁剪后的图,扩16个像素   

        float scale = cv::min(float(input_w) / (box.bbox.width+32), float(input_h) / (box.bbox.height+32));  //扩16个像素
        auto scaleSize = cv::Size((box.bbox.width+32) * scale, (box.bbox.height+32) * scale);

        cv::Mat resized;
        cv::resize(image_crop, resized, scaleSize, 0, 0);     
        cv::Mat cropped = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Rect rect((input_w - scaleSize.width) / 2, (input_h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);

        resized.copyTo(cropped(rect));

        // cv::Mat cropped = cv::imread("/home/qzx/code/SIL/test/test-python.png");

        constexpr static float mean[] = {0.485, 0.456, 0.406};
        constexpr static float std[] = {0.229, 0.224, 0.225};

        int row = inputTensorShape[1];
        int col = inputTensorShape[0];

        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    float pix = cropped.ptr<uchar>(i)[j * 3 + c];
                    output[c * row * col + i * col + j] = (pix / 255. - mean[c]) / std[c];

                                    
                }
            }
        }
                
    }

    void PreProcess::letterbox(const cv::Mat& image, cv::Mat& outImage,
                              const cv::Size& newShape = cv::Size(640, 640),
                              const cv::Scalar& color = cv::Scalar(114, 114, 114),
                              bool auto_ = true,
                              bool scaleFill = false,
                              bool scaleUp = true,
                              int stride = 32)
    {
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
        if (!scaleUp)
            r = std::min(r, 1.0f);

        float ratio[2] {r, r};
        int newUnpad[2] {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

        auto dw = (float)(newShape.width - newUnpad[0]);
        auto dh = (float)(newShape.height - newUnpad[1]);

        if (auto_)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            newUnpad[0] = newShape.width;
            newUnpad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
        {
            cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
        }

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }
}