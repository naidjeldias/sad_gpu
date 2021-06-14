#include <disparity.hpp>

double compute_disparity ( const cv::Mat &im_l, const cv::Mat &im_r, const int &win_size,
    const int &disp_range, cv::Mat disp_map
)
{
    double time = 0.0;
    for (int x = win_size/2; x < im_l.rows - win_size/2; x++){
        for(int y = win_size/2; y < im_l.cols - win_size/2; y++){
            int min_sad     = INT_MAX;
            int disp_value  = 0.0;
            for(int d = 0; d < disp_range; d++)
            {
                if((y + d) >= (im_l.cols - win_size/2))
                    break;
                int sad_value = compute_sad(im_l, im_r, win_size, cv::Point2i(y,x), cv::Point2i(y-d,x));
                if (sad_value < min_sad)
                {
                    min_sad     = sad_value;
                    disp_value  = d;
                }
            }
            disp_map.at<uchar>(x,y) = disp_value;
        }
    }
    return time;
}

int compute_sad (const cv::Mat &im_l, const cv::Mat &im_r, const int &win_size,
    const cv::Point2i p_l, const cv::Point2i p_r)
{
    int start = -(win_size )/2;
	int stop  = win_size -1;
	int sad_value = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
			sad_value += abs((int)im_l.at<uchar>(p_l.y + j,p_l.x + i) - (int)im_r.at<uchar>(p_r.y + j,p_r.x + i));
	return sad_value;
}