#include <opencv2/highgui.hpp>
#include <iostream>

#include <disparity.hpp>
#include <disparity.cuh>

int main(int argc, const char* argv[])
{	
	if (argc <= 2)
	{
		std::cout << "Usage: disparity <WIN_SIZE> <MAX_DISP_RANGE>" << std::endl;
		return 0;
	}

	unsigned int win_size 	= std::atoi(argv[1]);
	unsigned int max_range 	= std::atoi(argv[2]);	

	cv::Mat disp_map;
    cv::Mat im_left     = cv::imread("images/im2.ppm", cv::IMREAD_GRAYSCALE);
    cv::Mat im_right    = cv::imread("images/im6.ppm", cv::IMREAD_GRAYSCALE);

	cv::cuda::GpuMat im_left_gpu  (im_left);
    cv::cuda::GpuMat im_right_gpu (im_right);

    if(im_left.empty() || im_right.empty())
	{
		std::cerr << "could not load images." << std::endl;
		return -1;
	}

	std::cout << "computing disparity with win size "<< win_size << " and max disparity equal to "<< max_range << std::endl;
	
	disp_map 		= cv::Mat(im_left.rows, im_left.cols, CV_8UC1, cv::Scalar::all(0));
	
	cv::Mat  		disp_map_gpu 		(im_left.size(), im_left.type(), cv::Scalar::all(0));
	cv::cuda::GpuMat  d_disp_map_gpu 	(im_left.size(), im_left.type(), cv::Scalar::all(0));

	double cpu_time = compute_disparity (im_left, im_right, win_size, max_range, disp_map);
	std::cout << "time elapsed on cpu: " << cpu_time <<" ms" << std::endl;
	double gpu_time = compute_disparity_gpu(im_left_gpu, im_right_gpu, win_size, max_range, d_disp_map_gpu);
	std::cout << "time elapsed on gpu: " << gpu_time <<" ms" << std::endl;
	
	cv::imshow("left", im_left);
	cv::imshow("right", im_right);

	cv::medianBlur(disp_map,disp_map,3);
	cv::normalize(disp_map,disp_map,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::imshow("disparity", disp_map);

	d_disp_map_gpu.download(disp_map_gpu);
	cv::medianBlur(disp_map_gpu,disp_map_gpu,3);
	cv::normalize(disp_map_gpu,disp_map_gpu,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::imshow("disparity_gpu", disp_map_gpu);
	
	cv::waitKey(0);
	cv::destroyAllWindows();

    return 0;
}