#include <disparity.cuh>
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
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    const int stride_x  = blockDim.x * gridDim.x;
    const int stride_y  = blockDim.y * gridDim.y;

    int min_sad     = INT_MAX;
    int disp_value  = 0.0;

    for (int x = idx; x < im_l.cols; x+= stride_x)
    {
        for (int y = idy; y < im_l.rows; y+= stride_y)
        {
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
    }
    
    
}


double compute_disparity_gpu (const cv::Mat &im_left, const cv::Mat &im_right,
    const int &win_size, const int &disp_range, cv::Mat &disp_map)
{
    int deviceId;
    cudaGetDevice(&deviceId);

    cv::cuda::GpuMat im_l   (im_left.size(), CV_8UC1);
    im_l.upload (im_left);
    cv::cuda::GpuMat im_r   (im_right.size(), CV_8UC1);
    im_r.upload(im_right);

    void *disp_map_ptr;
	unsigned int frameByteSize = im_left.rows * im_left.cols;
	cudaMallocManaged(&disp_map_ptr, frameByteSize);
    cv::cuda::GpuMat  	d_disp_map_gpu 	(im_left.size(), CV_8UC1, disp_map_ptr);

    cudaMemPrefetchAsync(disp_map_ptr, frameByteSize, deviceId);

    const dim3 threadsPerBlock(32, 32);
	const dim3 blocksPerGrid(cv::cudev::divUp(disp_map.cols, threadsPerBlock.x), 
                        cv::cudev::divUp(disp_map.rows, threadsPerBlock.y));

    auto start = std::chrono::steady_clock::now();
    
    compute_sad<<<blocksPerGrid, threadsPerBlock>>>(im_l, im_r, win_size, disp_range, d_disp_map_gpu);
    
    CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaMemPrefetchAsync(disp_map_ptr, frameByteSize, cudaCpuDeviceId);
    //Copy data from device to host
	d_disp_map_gpu.download(disp_map);

    return time;
}