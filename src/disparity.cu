#include <disparity.cuh>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>

__device__ int compute_sad(unsigned char *im_l, unsigned char *im_r,
            const int &win_size, const int x, const int y, const int d, const int width)
{
    int start = -(win_size )/2;
	int stop  = win_size -1;
	int sad_value = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
            sad_value += abs((int)im_l[(y+j) * width + (x+i)] - (int)im_r[(y+j) * width + (x-d+i)]);
	return sad_value;
}
__global__ 
void compute_sad (unsigned char *im_l, unsigned char *im_r,
                const int win_size, const int disp_range, unsigned char *disp_map, const int width,
    const int height)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    const int stride_x  = blockDim.x * gridDim.x;
    const int stride_y  = blockDim.y * gridDim.y;

    int min_sad     = INT_MAX;
    int disp_value  = 0.0;

    for (int x = idx; x < width; x+= stride_x)
    {
        for (int y = idy; y < height; y+= stride_y)
        {
            if(y >= win_size/2 && y < height - win_size/2)
            {
                if(x >= win_size/2 && x < width - win_size/2)
                {
                    for(int d = 0; d < disp_range; d++)
                    {
                        if((x + d) >= (width - win_size/2))
                            break;
                        int sad_value = compute_sad(im_l, im_r, win_size, x, y, d, width);
                        if (sad_value < min_sad)
                        {
                            min_sad     = sad_value;
                            disp_value  = d;
                        }                    
                    }
                    disp_map[y * width + x] = (uchar) disp_value;
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
    unsigned int frameByteSize = im_left.rows * im_left.cols;

    cudaError_t dispMapErr;
    cudaError_t asyncErr;

    cudaDeviceSynchronize(); //this synchronize call is just to initialize the cuda runtime
    
    auto start = std::chrono::steady_clock::now();

    //Input 
    unsigned char *im_letf_ptr;
    cudaMalloc(&im_letf_ptr, frameByteSize);
    unsigned char *im_right_ptr;
    cudaMalloc(&im_right_ptr, frameByteSize);

    //Output
    unsigned char *disp_map_ptr;	
	cudaMalloc(&disp_map_ptr, frameByteSize);
    
    //Prefetching data
    cudaMemPrefetchAsync(im_letf_ptr, frameByteSize, deviceId);
    cudaMemPrefetchAsync(im_right_ptr, frameByteSize, deviceId);
    cudaMemPrefetchAsync(disp_map_ptr, frameByteSize, deviceId);
   
    //Copying data
    cudaMemcpy(im_letf_ptr, im_left.ptr(), frameByteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(im_right_ptr, im_right.ptr(), frameByteSize, cudaMemcpyHostToDevice);
    cudaMemset(disp_map_ptr, 0, frameByteSize*sizeof(uchar));


    const dim3 threadsPerBlock(32, 32);
	const dim3 blocksPerGrid(cv::cudev::divUp(disp_map.cols, threadsPerBlock.x), 
                        cv::cudev::divUp(disp_map.rows, threadsPerBlock.y));
    

    compute_sad<<<blocksPerGrid, threadsPerBlock>>>(im_letf_ptr, im_right_ptr, win_size, disp_range, disp_map_ptr, disp_map.cols, disp_map.rows);
    
    dispMapErr = cudaGetLastError();
      if(dispMapErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(dispMapErr));
  
    asyncErr = cudaDeviceSynchronize();
      if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    //Copy data from device to host
    cudaMemPrefetchAsync(disp_map_ptr, frameByteSize, cudaCpuDeviceId);
    cudaMemcpy(disp_map.ptr(), disp_map_ptr, frameByteSize, cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaFree(im_letf_ptr);
    cudaFree(im_right_ptr);
    cudaFree(disp_map_ptr);

    return time;
}