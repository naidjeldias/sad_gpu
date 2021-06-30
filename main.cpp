#include <opencv2/highgui.hpp>
#include <iostream>

#include <disparity.hpp>
#include <disparity.cuh>

int main(int argc, const char* argv[])
{	
	if (argc <= 2)
	{
		std::cout << "Usage: disparity <WIN_SIZE> <MAX_DISP_RANGE> [<RESOLUTION> (F - full size; H - Half size)]" << std::endl;
		return 0;
	}

	unsigned int win_size 	= std::atoi(argv[1]);
	unsigned int max_range 	= std::atoi(argv[2]);

	cv::Mat im_left, im_right;
	//TO DO: image padding
	//Input
	if (argc == 3)
	{
		im_left     = cv::imread("images/im2.ppm", cv::IMREAD_GRAYSCALE);
    	im_right    = cv::imread("images/im6.ppm", cv::IMREAD_GRAYSCALE);
	}else
	{	
		std::string resolution 	= argv[3];	
		std::string left_file = "images/im2"+resolution+".ppm";
		std::string right_file = "images/im6"+resolution+".ppm";
		im_left     = cv::imread(left_file, cv::IMREAD_GRAYSCALE);
    	im_right    = cv::imread(right_file, cv::IMREAD_GRAYSCALE);
	}



	//Output
	cv::Mat  disp_map 		= cv::Mat(im_left.rows, im_left.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat  disp_map_gpu	= cv::Mat(im_left.rows, im_left.cols, CV_8UC1);
	
    if(im_left.empty() || im_right.empty())
	{
		std::cerr << "could not load images." << std::endl;
		return -1;
	}

	std::cout << "computing disparity with win size "<< win_size << " and max disparity equal to "<< max_range << "image resolution "<< im_left.cols << " X " << im_left.rows <<std::endl;
	double sum_time_gpu = 0.0, sum_time_cpu = 0.0;
	for (int i = 0; i < 5; i++)
	{
		double cpu_time = compute_disparity (im_left, im_right, win_size, max_range, disp_map);
		sum_time_cpu += cpu_time;
		double gpu_time = compute_disparity_gpu(im_left, im_right, win_size, max_range, disp_map_gpu);
		sum_time_gpu += gpu_time;

	}
	double mean_time_cpu = sum_time_cpu/5;
	double mean_time_gpu = sum_time_gpu/5;
	std::cout << "Mean time elapsed on cpu: " << mean_time_cpu <<" ms" << std::endl;
	std::cout << "Mean time elapsed on gpu: " << mean_time_gpu <<" ms" << std::endl;

	std::cout << "SpeedUp (Tcpu/Tgpu): " << mean_time_cpu / mean_time_gpu << std::endl;
	
	cv::imshow("left", im_left);
	cv::imshow("right", im_right);

	//Normalization for visualization purpose
	cv::medianBlur(disp_map,disp_map,3);
	cv::normalize(disp_map,disp_map,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::imshow("disparity", disp_map);
	
	//Normalization for visualization purpose
	cv::medianBlur(disp_map_gpu,disp_map_gpu,3);
	cv::normalize(disp_map_gpu,disp_map_gpu,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::imshow("disparity_gpu", disp_map_gpu);

	// cv::imwrite("gpu_map.png", disp_map_gpu);
	// cv::imwrite("cpu_map.png", disp_map);
	
	cv::waitKey(0);
	cv::destroyAllWindows();

    return 0;
}