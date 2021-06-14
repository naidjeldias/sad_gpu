#include <disparity.hpp>

double compute_disparity ( const cv::Mat &im_l, const cv::Mat &im_r, const int &win_size,
    const int &disp_range, cv::Mat disp_map
)
{
    auto start = std::chrono::steady_clock::now();
    for (int y = win_size/2; y < im_l.rows - win_size/2; y++){
        for(int x = win_size/2; x < im_l.cols - win_size/2; x++){
            int min_sad     = INT_MAX;
            int disp_value  = 0.0;
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
            disp_map.at<uchar>(y,x) = disp_value;
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return time;
}

int compute_sad (const cv::Mat &im_l, const cv::Mat &im_r, const int &win_size,
    const int x, const int y, const int d)
{
    int start = -(win_size )/2;
	int stop  = win_size -1;
	int sad_value = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
			sad_value += abs((int)im_l.at<uchar>(y + j, x + i) - (int)im_r.at<uchar>(y + j, x - d + i));
	return sad_value;
}