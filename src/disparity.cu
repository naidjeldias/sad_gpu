#include <disparity.cuh>
// #include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>

__device__ int compute_sad(const cv::cudev::PtrStepSz<uchar> im_l, const cv::cudev::PtrStepSz<uchar> im_r,
            const int &win_size, const int x, const int y, const int d)
{
    int start = -(win_size )/2;
	int stop  = win_size -1;
	int sad_value = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
			sad_value += abs((int)im_l.ptr(y + j)[x + i] - (int)im_r.ptr(y + j)[x - d + i]);
	return sad_value;
}
__global__ 
void compute_sad (const cv::cudev::PtrStepSz<uchar> im_l, const cv::cudev::PtrStepSz<uchar> im_r,
                const int win_size, const int disp_range, cv::cudev::PtrStepSz<uchar> disp_map)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int min_sad     = INT_MAX;
    int disp_value  = 0.0;

    if(y >= win_size/2 && y < im_l.rows - win_size/2)
    {
        if(x >= win_size/2 && x < im_l.cols - win_size/2)
        {
            for(int d = 0; d < disp_range; d++)
            {
                if((x + d) >= (im_l.cols - win_size/2))
                    break;
                int sad_value = compute_sad(im_l, im_r, win_size, x, y, d);
                if (sad_value < min_sad)
                {
                    min_sad     = sad_value;
                    disp_value  = d;
                }                    
            }
            disp_map.ptr(y)[x] = (uchar) disp_value;
        }
    }
    
}


double compute_disparity_gpu (const cv::cuda::GpuMat &im_l, const cv::cuda::GpuMat &im_r,
    const int &win_size, const int &disp_range, cv::cuda::GpuMat &disp_map)
{
    const dim3 block(64, 2);
	const dim3 grid(cv::cudev::divUp(disp_map.cols, block.x), cv::cudev::divUp(disp_map.rows, block.y));
    
    auto start = std::chrono::steady_clock::now();
    compute_sad<<<grid, block>>>(im_l, im_r, win_size, disp_range, disp_map);
    CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return time;
}