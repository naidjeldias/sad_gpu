#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

double compute_disparity
(
    const cv::Mat &im_l,
    const cv::Mat &im_r,
    const int &win_size,
    const int &disp_range,
    cv::Mat disp_map
);

int compute_sad
(
    const cv::Mat &im_l,
    const cv::Mat &im_r,
    const int &win_size,
    const int x,
    const int y,
    const int d
);