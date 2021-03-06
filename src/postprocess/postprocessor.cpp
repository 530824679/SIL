#include "postprocessor.h"
#include "visualization.h"
#include <algorithm>
using namespace perception;
Visualizer visualizer;
namespace perception
{
    std::vector<BoxInfo> PostProcess::postprocessing2D(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, std::vector<Ort::Value>& outputTensors, const float& confThreshold, const float& iouThreshold)
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;

        auto* rawOutput = outputTensors[0].GetTensorData<float>();
        std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output(rawOutput, rawOutput + count);

        for (const int64_t& shape : outputShape)
            std::cout << "Output Shape: " << shape << std::endl;

        // first 5 elements are box[4] and obj confidence
        int numClasses = (int)outputShape[2] - 5;
        int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

        // only for batch size = 1
        for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
        {
            float clsConf = it[4];

            if (clsConf > confThreshold)
            {
                if (it[2] >10 and it[3]>5)
                {
                    int centerX = (int) (it[0]);
                    int centerY = (int) (it[1]);
                    int width = (int) (it[2]);
                    int height = (int) (it[3]);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    float objConf;
                    int classId;
                    this->getBestClassInfo(it, numClasses, objConf, classId);

                    float confidence = clsConf * objConf;

                    boxes.emplace_back(left, top, width, height);
                    confs.emplace_back(confidence);
                    classIds.emplace_back(classId);
                }
            }
        }

        std::vector<int> indices;
        nmsBoxes(boxes, confs, confThreshold, iouThreshold, indices);
        std::cout << "amount of NMS indices: " << indices.size() << std::endl;

        std::vector<BoxInfo> detections;

        for (int idx : indices)
        {
            BoxInfo det;
            det.bbox = cv::Rect(boxes[idx]);
            scaleCoords(resizedImageShape, det.bbox, originalImageShape);

            det.score = confs[idx];
            det.classId = classIds[idx];
            detections.emplace_back(det);
        }

        return detections;
    }


    void PostProcess::getMaxPreds(const float* heatmap, std::vector<int64_t>& dim, std::vector<float> preds, std::vector<float> maxvals, std::vector<KeypointsInfo> &KPS,BoxInfo box) 
    {
        float scale_w = 256 / (box.bbox.width);
        float scale_h = 256 / (box.bbox.height);
        int num_joints = dim[1];
        int width = dim[3];
        std::vector<int> idx;
        idx.resize(num_joints * 2);
        KeypointsInfo temp;

        for (int j = 0; j < 4; j++) 
        {
            const float* index = &(heatmap[0 * num_joints * dim[2] * dim[3] + j * dim[2] * dim[3]]);
            const float* end = index + dim[2] * dim[3];
            const float* max_dis = std::max_element(index, end);
            auto max_id = std::distance(index, max_dis);
            maxvals[j] = *max_dis;
            if (*max_dis > 0)
             {
                preds[j * 2] = static_cast<float>(max_id % width) * 4+2 ;
                // std::cout << "x: " << preds[j * 2] << std::endl;
                preds[j * 2 + 1] = static_cast<float>(max_id / width) * 4+2 ;
                // std::cout << "y: " << preds[j * 2 + 1] << std::endl;

            }
        }


        temp.pt1.x = (preds[0]/256*(box.bbox.width+32)) + box.bbox.x-16;
        temp.pt1.y = (preds[1]/192*(box.bbox.height+32)) + box.bbox.y-16;
        temp.pt2.x = (preds[2]/256*(box.bbox.width+32)) + box.bbox.x-16;
        temp.pt2.y = (preds[3]/192*(box.bbox.height+32)) + box.bbox.y-16;
        temp.pt3.x = (preds[4]/256*(box.bbox.width+32)) + box.bbox.x-16;
        temp.pt3.y = (preds[5]/192*(box.bbox.height+32)) + box.bbox.y-16;
        temp.pt4.x = (preds[6]/256*(box.bbox.width+32)) + box.bbox.x-16;
        temp.pt4.y = (preds[7]/192*(box.bbox.height+32)) + box.bbox.y-16;

        KPS.push_back(temp);



    }


    // void PostProcess::getMaxPreds(const float* heatmap, std::vector<int64_t>& t, std::vector<float> preds, std::vector<float> maxvals, std::vector<KeypointsInfo> &KPS)
    // {
    //     int batch_size = t[0];
    //     int num_joints = t[1];
    //     int width = t[3];


    //     float* pred_mask = new float[num_joints * 2];
    //     int* idx = new int[num_joints * 2];
    //     for (int i = 0; i < batch_size; ++i) {
    //         for (int j = 0; j < num_joints; ++j) {
    //             float max = heatmap[i * num_joints * t[2] * t[3] + j * t[2] * t[3]];
    //             // std::cout << max<< std::endl;
    //             int max_id = 0;
    //             for (int k = 1; k < t[2] * t[3]; ++k) {
    //                 int index = i * num_joints * t[2] * t[3] + j * t[2] * t[3] + k;
    //                 if (heatmap[index] > max) {
    //                     max = heatmap[index];
    //                     max_id = k;
    //                 }
    //             }
    //             maxvals[j] = max;
    //             idx[j] = max_id;
    //             idx[j + num_joints] = max_id;
    //         }
    //     }

    //     for (int i = 0; i < num_joints; ++i) {
    //         printf("num joints is %d\t i is %d\n", num_joints, i);
    //         idx[i] = idx[i] % width;
    //         idx[i + num_joints] = idx[i + num_joints] / width;
    //         if (maxvals[i] > 0) {
    //             pred_mask[i] = 1.0;
    //             pred_mask[i + num_joints] = 1.0;
    //         }
    //         else {
    //             pred_mask[i] = 0.0;
    //             pred_mask[i + num_joints] = 0.0;
    //         }
    //         preds[i] = idx[i] * pred_mask[i];
    //         preds[i + num_joints] = idx[i + num_joints] * pred_mask[i + num_joints];

    //         std::cout <<"###################"<< std::endl;
    //         std::cout << preds[i]<< std::endl;
    //         std::cout << preds[i+ num_joints]<< std::endl;
    //         std::cout <<"$$$$$$$$$$$$$$$$$$$"<< std::endl;

    //     }
    //     KPS[0].pt1.x = preds[0];
    //     KPS[0].pt1.y = preds[1];
    //     KPS[0].pt2.x = preds[2];
    //     KPS[0].pt2.y = preds[3];
    //     KPS[0].pt3.x = preds[4];
    //     KPS[0].pt3.y = preds[5];
    //     KPS[0].pt4.x = preds[6];
    //     KPS[0].pt4.y = preds[7];
    //     printf("enter out!");
    // }

    std::vector<KeypointsInfo> PostProcess::postprocessing3D(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, std::vector<Ort::Value>& outputTensors,std::vector<int64_t>& t,BoxInfo box)
    {   
        
        std::vector<KeypointsInfo> KPS1;
        std::vector<float> preds(t[1] * 2, 0);
        std::vector<float> maxvals(t[1], 0);
        auto* rawOutput = outputTensors[0].GetTensorData<float>();
        size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        // std::cout << count << std::endl;
        getMaxPreds(rawOutput, t, preds, maxvals,KPS1,box);
        return KPS1;



        // return xxxx
    }

    bool comp(BoxInfo box1,BoxInfo box2)
    {
        return (box1.score > box2.score);

    }

    void PostProcess::nmsBoxes(std::vector<cv::Rect>& boxes, std::vector<float>& confs, const float& confThreshold, const float& iouThreshold, std::vector<int>& indices)
    {
        BoxInfo bbox;
        std::vector<BoxInfo> bboxes;
        int i, j;
        for (i = 0; i < boxes.size(); i++)
        {
            bbox.bbox = boxes[i];
            bbox.score = confs[i];
            bbox.classId = i;
            bboxes.push_back(bbox);
        }
        sort(bboxes.begin(), bboxes.end(), comp);

        int updated_size = bboxes.size();
        for (i = 0; i < updated_size; i++)
        {
            if (bboxes[i].score < confThreshold)
                continue;
            indices.push_back(bboxes[i].classId);
            for (j = i + 1; j < updated_size; j++)
            {
                float iou = get_iou_value(bboxes[i].bbox, bboxes[j].bbox);
                if (iou > iouThreshold)
                {
                    bboxes.erase(bboxes.begin() + j);
                    j = j - 1;
                    updated_size = bboxes.size();
                }
            }
        }
    }

    void PostProcess::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId)
    {
        // first 5 element are box and obj confidence
        bestClassId = 5;
        bestConf = 0;

        for (int i = 5; i < numClasses + 5; i++)
        {
            if (it[i] > bestConf)
            {
                bestConf = it[i];
                bestClassId = i - 5;
            }
        }

    }


    void PostProcess::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
    {
        float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                              (float)imageShape.width / (float)imageOriginalShape.width);

        int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                      (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

        coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
        coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

        coords.width = (int) std::round(((float)coords.width / gain));
        coords.height = (int) std::round(((float)coords.height / gain));

        // // clip coords, should be modified for width and height
        // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
        // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
        // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
        // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
    }

    float PostProcess::get_iou_value(cv::Rect rect1, cv::Rect rect2)
    {
        int xx1, yy1, xx2, yy2;

        xx1 = std::max(rect1.x, rect2.x);
        yy1 = std::max(rect1.y, rect2.y);
        xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

        int insection_width, insection_height;
        insection_width = std::max(0, xx2 - xx1 + 1);
        insection_height = std::max(0, yy2 - yy1 + 1);

        float insection_area, union_area, iou;
        insection_area = float(insection_width) * insection_height;
        union_area = float(rect1.width*rect1.height + rect2.width*rect2.height - insection_area);
        iou = insection_area / union_area;
        return iou;
    }
}
