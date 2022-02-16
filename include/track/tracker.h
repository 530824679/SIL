#ifndef __TRACK_HPP__
#define __TRACK_HPP__

#include <set>
#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define CNUM 255 // max num. of people per frame


class Track
{
public:
    Track() = default;

    Track(int max_age);

    ~Track(){}

    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

    vector<TrackingBox> update(const vector<TrackingBox> &detFrameData);

private:

    int total_frames = 0;
    double total_time = 0.0;

    Scalar_<int> randColor[CNUM];

    int frame_count = 0;
    int max_age = 15;
    int min_hits = 3;
    double iouThreshold = 0.2;
    vector<KalmanTracker> trackers;

    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;
};

#endif