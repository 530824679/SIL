#include "preprocessor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#define NUM_FRAME 300
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

    cv::Mat PreProcess::data_input(const std::string& filename)
    {
        cv::Mat frame;
        std::cout <<  "input image or video filename" << std::endl;

        const std::string  file_ext = filename.substr(filename.find_last_of(".") + 1);
        const std::string  protocol = filename.substr(0, 7);
        //如果是图像
        if (file_ext == "png")
	    {
            cv::Mat frame = imread(filename);
        }
        else
        {
            cv::VideoCapture capture(filename);
            if (!capture.isOpened())
            {
                std::cout << "open video error"<< std::endl;;
            }
                /*CV_CAP_PROP_POS_MSEC – 视频的当前位置（毫秒）
                CV_CAP_PROP_POS_FRAMES – 视频的当前位置（帧）
                CV_CAP_PROP_FRAME_WIDTH – 视频流的宽度
                CV_CAP_PROP_FRAME_HEIGHT – 视频流的高度
                CV_CAP_PROP_FPS – 帧速率（帧 / 秒）*/
            int frame_width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
            int frame_height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
            float frame_fps = capture.get(CV_CAP_PROP_FPS);
            int frame_number = capture.get(CV_CAP_PROP_FRAME_COUNT);//总帧数
            std::cout << "frame_width is " << frame_width<< std::endl;
            std::cout << "frame_height is " << frame_height << std::endl;
            std::cout << "frame_fps is " << frame_fps << std::endl;
            std::cout << "frame_number is " << frame_number << std::endl;

            while (true)
            {
                
                //从视频中读取一个帧
                frame = capture.read(frame);
                imshow("MyVideo", frame);

                // bool bSuccess = capture.read(frame);
                
                // if (!bSuccess)
                // {
                //     std::cout << "Cannot read frames from video file" << std::endl;
                //     break;
                // }
                // return frame; 
            }
        
        }
        // return frame; 
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
        // 抠图box
        
        cv::Mat image_crop = image(cv::Rect(box.bbox)); // 裁剪后的图
        // cv::Mat image_crop = cv::imread("/home/qzx/code/SIL/test/00000031.png");
        
        

        int input_w = inputTensorShape[0];
        int input_h = inputTensorShape[1];
        float scale = cv::min(float(input_w) / image.cols, float(input_h) / image.rows);
        auto scaleSize = cv::Size(image.cols * scale, image.rows * scale);
        // auto scaleSize = cv::Size(256, 192);

        cv::Mat resized;
        cv::resize(image_crop, resized, scaleSize, 0, 0);
        cv::imwrite("/home/qzx/code/SIL/test/test.png",resized);       
        cv::Mat cropped = cv::Mat::zeros(input_h, input_w, CV_8UC3);
        cv::Rect rect((input_w - scaleSize.width) / 2, (input_h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);

        resized.copyTo(cropped(rect));

        cv::imwrite("/home/qzx/code/SIL/test/test1.png",cropped);
        

        constexpr static float mean[] = {0.5194416012442385, 0.5378052387430711, 0.533462090585746};
        constexpr static float std[] = {0.3001546018824507, 0.28620901391179554, 0.3014112676161966};

        int row = inputTensorShape[1];
        // std::cout << inputTensorShape[1] << std::endl;
        int col = inputTensorShape[0];
        // std::cout << inputTensorShape[0] << std::endl;

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