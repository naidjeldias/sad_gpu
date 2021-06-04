#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, const char* argv[])
{
    cv::Mat im_left     = cv::imread("images/im0.ppm", cv::IMREAD_GRAYSCALE);
    cv::Mat im_right    = cv::imread("images/im1.ppm", cv::IMREAD_GRAYSCALE);

    if(im_left.empty() || im_right.empty())
	{
		std::cerr << "could not load images." << std::endl;
		return -1;
	}

    cv::imshow("left", im_left);
	cv::imshow("right", im_right);

	cv::waitKey(0);
	cv::destroyAllWindows();

    return 0;
}