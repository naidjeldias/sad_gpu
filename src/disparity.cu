#include <disparity.cuh>
// #include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>

__global__ 
void compute_sad (const cv::cudev::PtrStepSz<uchar> im_l, const cv::cudev::PtrStepSz<uchar> im_r,
                const int win_size, const int disp_range, cv::cudev::PtrStepSz<uchar> disp_map)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= win_size/2 && x < im_l.rows - win_size/2){
        if(y >= win_size/2 && y < im_l.cols - win_size/2){
            
        }
    }
}


double compute_disparity_gpu (const cv::cuda::GpuMat &im_l, const cv::cuda::GpuMat &im_r,
    const int &win_size, const int &disp_range, cv::cuda::GpuMat &disp_map)
{
    double time = 0.0;
    compute_sad<<<1,1>>>(im_l, im_r, win_size, disp_range, disp_map);
    return time;
}