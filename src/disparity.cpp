#include <disparity.hpp>

double compute_disparity ( const cv::Mat &im_l, const cv::Mat &im_r, const int &win_size,
    const int &disp_range, cv::Mat disp_map
)
{
    double time = 0.0;
    for (int i = win_size/2; i < im_l.rows - win_size/2; i++){
        for(int j = win_size/2; j < im_l.cols - win_size/2; j++){
            int min_sad     = INT_MAX;
            int disp_value  = 0.0;
            for(int k = 0; k < disp_range; k++)
            {
                if((j + k) >= (im_l.cols - win_size/2))
                    break;
                int sad_value = compute_sad(im_l, im_r, win_size, cv::Point2i(j,i), cv::Point2i(j-k,i));
                if (sad_value < min_sad)
                {
                    min_sad     = sad_value;
                    disp_value  = k;
                }
            }
            disp_map.at<uchar>(i,j) = disp_value;
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