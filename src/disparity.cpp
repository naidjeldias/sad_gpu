#include <disparity.hpp>
/**
* Cria um mapa de disparidade de duas imagens de um par estéreo
*
* @param[in] im_l endereço da matriz contendo os pixels da imagem esquerda da câmera estéreo
* @param[in] im_r endereço da matriz contendo os pixels da imagem direita da câmera estéreo
* @param[in] win_size dimensao da janela de vizinhança considerando uma janela quadrada [win_size X win_size]
* @param[in] disp_range nível de disparidade máxima 
* @param[out] dis_map  matriz contendo o resultado da disparidade encontrada para cada pixel 
*/
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
                if((x - d) < win_size/2)
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

/**
* Calcula o valor de SAD para um determinado pixel com base na sua vizinhança
*
* @param[in] im_l endereço da matriz contendo os pixels da imagem esquerda da câmera estéreo
* @param[in] im_r endereço da matriz contendo os pixels da imagem direita da câmera estéreo
* @param[in] win_size dimensao da janela de vizinhança considerando uma janela quadrada [win_size X win_size]
* @param[in] x índice da coluna do pixel em questão 
* @param[in] y índice da linha do pixel em questão
* @param[in] d nível de disparidade a qual quer se calcular o valor de SAD
* @param[out] sad_value resultado da métrica SAD
*/
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