#include <disparity.cuh>
#include <cuda_runtime.h>

/**
* Calcula o valor de SAD para um determinado pixel com base na sua vizinhança
*
* @param[in] im_l ponteiro do vetor contendo os pixels da imagem esquerda da câmera estéreo
* @param[in] im_r ponteiro do vetor contendo os pixels da imagem direita da câmera estéreo
* @param[in] win_size dimensao da janela de vizinhança considerando uma janela quadrada [win_size X win_size]
* @param[in] x índice da coluna do pixel em questão
* @param[in] y índice da linha do pixel em questão
* @param[in] width largura da imagem para realizar o acesso a imagem vetorizada
* @param[in] d nível de disparidade a qual quer se calcular o valor de SAD
* @param[out] sad_value resultado da métrica SAD
*/
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

/**
* Cria um mapa de disparidade de duas imagens de um par estéreo
*
* @param[in] im_l ponteiro do vetor contendo os pixels da imagem esquerda da câmera estéreo
* @param[in] im_r ponteiro do vetor contendo os pixels da imagem direita da câmera estéreo
* @param[in] win_size dimensao da janela de vizinhança considerando uma janela quadrada [win_size X win_size]
* @param[in] disp_range nível de disparidade máxima 
* @param[in] width largura da imagem para realizar o acesso a imagem vetorizada
* @param[out] dis_map ponteiro do vetor contendo o resultado da disparidade encontrada para cada pixel 
*/

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
                        if((x - d) < win_size/2)
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
    unsigned int frame_size = im_left.rows * im_left.cols;

    cudaError_t dispMapErr;
    cudaError_t asyncErr;

    cudaDeviceSynchronize(); //this synchronize call is just to initialize the cuda runtime
    
    auto start = std::chrono::steady_clock::now();

    //Input 
    unsigned char *im_letf_ptr;
    cudaMallocManaged(&im_letf_ptr, frame_size*sizeof(uchar));
    unsigned char *im_right_ptr;
    cudaMallocManaged(&im_right_ptr, frame_size*sizeof(uchar));

    cudaMemPrefetchAsync(im_letf_ptr, frame_size, deviceId);
    cudaMemPrefetchAsync(im_right_ptr, frame_size, deviceId);

    //Output
    unsigned char *disp_map_ptr;	
	cudaMallocManaged(&disp_map_ptr, frame_size*sizeof(uchar));
    cudaMemPrefetchAsync(disp_map_ptr, frame_size, cudaCpuDeviceId);

    //Copying data
    cudaMemcpy(im_letf_ptr, im_left.ptr(), frame_size, cudaMemcpyHostToDevice);
    cudaMemcpy(im_right_ptr, im_right.ptr(), frame_size, cudaMemcpyHostToDevice);
    cudaMemset(disp_map_ptr, 0, frame_size*sizeof(uchar));

    const dim3 threadsPerBlock(16, 16);
	const dim3 blocksPerGrid(((disp_map.cols + threadsPerBlock.x - 1)/threadsPerBlock.x), 
                        ((disp_map.rows + threadsPerBlock.y - 1)/threadsPerBlock.y));
    

    compute_sad<<<blocksPerGrid, threadsPerBlock>>>(im_letf_ptr, im_right_ptr, win_size, disp_range, disp_map_ptr, disp_map.cols, disp_map.rows);
    
    dispMapErr = cudaGetLastError();
      if(dispMapErr != cudaSuccess) printf("Last error: %s\n", cudaGetErrorString(dispMapErr));
  
    asyncErr = cudaDeviceSynchronize();
      if(asyncErr != cudaSuccess) printf("Device synchrinize error: %s\n", cudaGetErrorString(asyncErr));

    //Copy data from device to host
    cudaMemcpy(disp_map.ptr(), disp_map_ptr, frame_size, cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    cudaFree(im_letf_ptr);
    cudaFree(im_right_ptr);
    cudaFree(disp_map_ptr);

    return time;
}
