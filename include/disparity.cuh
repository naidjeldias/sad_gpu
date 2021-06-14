#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>

double compute_disparity_gpu
(
    const cv::cuda::GpuMat &im_l,
    const cv::cuda::GpuMat &im_r,
    const int &win_size,
    const int &disp_range,
    cv::cuda::GpuMat &disp_map
);