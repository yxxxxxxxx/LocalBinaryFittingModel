#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <chrono>
#define pi 3.14159265
using namespace std;
using namespace chrono;

cv::Mat conv2(const cv::Mat &img, const cv::Mat &ikernel)
{
	cv::Mat dest;
	cv::Mat kernel;
	cv::flip(ikernel,kernel,-1);
	cv::Mat source = img;
	cv::Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
	int borderMode = cv::BORDER_CONSTANT;
	cv::filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode); 
	return dest;
}
cv::Mat NeumannBoundCond(cv::Mat &u)
{
	int w = u.cols - 1;
	int h = u.rows - 1;
	u.at<double>(0, 0) = u.at<double>(2, 2);
	u.at<double>(h, 0) = u.at<double>(h - 2, 2);
	u.at<double>(0, w) = u.at<double>(2, w - 2);
	u.at<double>(h, w) = u.at<double>(h - 2, w - 2);
	for(int i = 1; i <= w - 1 ; i++)
	{
		u.at<double>(0, i) = u.at<double>(2, i);
		u.at<double>(h, i) = u.at<double>(h - 2, i);
	}
	for(int i = 1; i <= h - 1 ; i++)
	{
		u.at<double>(i, 0) = u.at<double>(i, 2);
		u.at<double>(i, w) = u.at<double>(i, w - 2);
	}
	return u;
}
cv::Mat gradient_x(cv::Mat &input)
{
	cv::Mat Ix(input.size(), input.type());
	for (int ncol = 0; ncol < input.cols; ncol++)
	{
		for (int nrow = 0; nrow < input.rows; nrow++)
		{
			if (ncol == 0) 
			{
				Ix.at<double>(nrow, ncol) = input.at<double>(nrow, 1) - input.at<double>(nrow, 0);
			}
			else if (ncol == input.cols - 1) 
			{
				Ix.at<double>(nrow, ncol) = input.at<double>(nrow, ncol) - input.at<double>(nrow, ncol - 1);
			}
			else
				Ix.at<double>(nrow, ncol) = (input.at<double>(nrow, ncol + 1) - input.at<double>(nrow, ncol - 1)) / 2;
		}
	}
	return Ix;
}

cv::Mat gradient_y(cv::Mat &input)
{
	cv::Mat Iy(input.size(), input.type());
	for (int nrow = 0; nrow < input.rows; nrow++)
	{
		for (int ncol = 0; ncol < input.cols; ncol++)
		{
			if (nrow == 0) 
			{
				Iy.at<double>(nrow, ncol) = input.at<double>(1, ncol) - input.at<double>(0, ncol);
			}
			else if (nrow == input.rows - 1) 
			{
				Iy.at<double>(nrow, ncol) = input.at<double>(nrow, ncol) - input.at<double>(nrow - 1, ncol);
			}
			else
			Iy.at<double>(nrow, ncol) = (input.at<double>(nrow + 1, ncol) - input.at<double>(nrow - 1, ncol)) / 2;
		}
	}
	return Iy;
}
cv::Mat Matatan(cv::Mat u)
{
	cv::Mat dst(u.size(), u.type());
	for (int k = 0; k < u.rows; k++)
	{
		for (int i = 0; i < u.cols; i++)
		{
			dst.at<double>(k, i) = atan(u.at<double>(k, i));
		}
	}
	return dst;
}
cv::Mat curvature_central(cv::Mat &u)
{
	cv::Mat Ix, Iy;
	Ix = gradient_x(u);
	Iy = gradient_y(u);
	cv::Mat s;
	cv::magnitude(Ix, Iy, s);//梯度的模
	cv::Mat Nx = Ix / s;
	cv::Mat Ny = Iy / s;
	cv::Mat Nxx, Nyy;
	Nxx = gradient_x(Nx);
	Nyy = gradient_y(Ny);
	cv::Mat cur = Nxx + Nyy;
	return cur;
}

pair<cv::Mat, cv::Mat> LBF_LocalBinaryFit(cv::Mat K, cv::Mat &Img, cv::Mat &KI, cv::Mat &KONE, cv::Mat H)
{
	cv::Mat I = Img.mul(H);
	cv::Mat c1 = conv2(H, K);
	cv::Mat c2 = conv2(I, K);
	cv::Mat f1 = c2 / c1;
	cv::Mat f2 = (KI - c2) / (KONE - c1);
	return make_pair(f1, f2);
}

cv::Mat LBF_dataForce(cv::Mat &Img, cv::Mat &K, cv::Mat &KONE, cv::Mat f1, cv::Mat f2, int lambda1, int lambda2)
{
	cv::Mat s1 = lambda1 * f1.mul(f1) - lambda2 * f2.mul(f2);
	cv::Mat s2 = lambda1 * f1 - lambda2 * f2;
	cv::Mat f = ((lambda1 - lambda2) * KONE).mul(Img).mul(Img) + conv2(s1, K) - 2 * Img.mul(conv2(s2, K));
	return f;
}
cv::Mat EVOL_LBF(cv::Mat &u0, cv::Mat &Img, cv::Mat &K, cv::Mat &KI, cv::Mat &KONE, double nu, double timestep, double mu, int lambda1, int lambda2, 
	double epsilon, int numIter)
{
	cv::Mat u = u0;
	for(int i = 1; i <= numIter; i++)
	{
		u = NeumannBoundCond(u);
		cv::Mat C = curvature_central(u);
		cv::Mat HeavU = 0.5 * (1 + (2 / pi) * Matatan(u / epsilon));
		cv::Mat DiracU = (epsilon / pi) / (epsilon * epsilon+ u.mul(u)); 
		pair<cv::Mat, cv::Mat> f1_f2 = LBF_LocalBinaryFit(K, Img, KI, KONE, HeavU);
		cv::Mat LBF=LBF_dataForce(Img, K, KONE, f1_f2.first, f1_f2.second, lambda1, lambda2);
		cv::Mat areaTerm = -1 * DiracU.mul(LBF);
		cv::Mat Lap;
		cv::Laplacian(u, Lap, CV_64F);
		cv::Mat penalizeTerm = mu * (Lap-C);
		cv::Mat lengthTerm = nu * DiracU.mul(C);
		u = u + timestep * (lengthTerm + penalizeTerm + areaTerm);
	}
	return u;
} 
std::vector<std::vector<int> > SegmentMeasure(cv::Mat img)
{
	//Segment
	int boardsize = 20;
	int iscircle = 1;
	int r = 10;
	int centerx = 200; //可变参数，随机生成
	int centery = 200; //可变参数，随机生成
	cv::resize(img, img, cv::Size(400, 300));
	if(img.channels() == 3)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	}
	img.convertTo(img, CV_64F);
	cv::Mat phi(img.size(), CV_64F, cv::Scalar(-2));
	for(int x = 0; x < img.rows; x++)
	{
		for(int y = 0; y < img.cols; y++)
		{
			int d = (x - centerx) * (x - centerx) + (y - centery) * (y - centery);
			if (d <= r * r)
			{
				phi.at<double>(x, y) = 2;
			}
		}
	}
	int iterNum = 500;
	int lambda1 = 1;  
	int lambda2 = 1;
	double mu = 0.002 * 255 * 255;
	double pu = 1.0;
	double timestep = 0.1;
	double epsilon  = 1.0;
	// Guassian Kernel
	int sigmaG = 3;
	cv::Mat Klbf = cv::getGaussianKernel(4*sigmaG + 1, sigmaG, CV_64F);
	Klbf = Klbf * Klbf.t();
	cv::Mat KIG = conv2(img, Klbf);
	cv::Mat KONE = conv2(cv::Mat::ones(300, 400, CV_64F), Klbf);
	cv::Mat Kligf = Klbf;
	cv::Mat phi_LBF = phi;
	for(int iter = 1; iter <= iterNum; iter++)
	{
		int numIter = 1;
		phi_LBF = EVOL_LBF(phi_LBF, img, Klbf, KIG, KONE, mu, timestep, pu, lambda1, lambda2, epsilon, numIter);
	}	
	cv::Mat SaveLBF = cv::Mat::zeros(phi_LBF.size(), CV_8UC1);
	for(int x = 0; x < phi_LBF.rows; x++)
	{
		for(int y = 0; y < phi_LBF.cols; y++)
		{
			if (phi_LBF.at<double>(x, y) > 0)
			{
				SaveLBF.at<uchar>(x, y) = 255;
			}
		}
	}
	//Measure
	cv::Mat labels, stats, centers;
	int nccomps = cv::connectedComponentsWithStats (
            SaveLBF, //二值图像
            labels,     //和原图一样大的标记图
            stats, //nccomps×5的矩阵 表示每个连通区域的外接矩形和面积（pixel）
            centers //nccomps×2的矩阵 表示每个连通区域的质心
            );
	std::vector<std::vector<int> > result(nccomps);
	for(int iter = 0; iter < nccomps; iter++)
	{	
		int x = stats.at<int>(iter, 0);
  		int y = stats.at<int>(iter, 1);
  		int w = stats.at<int>(iter, 2);
  		int h = stats.at<int>(iter, 3);
  		int area = stats.at<int>(iter, 4);
  		result[iter].push_back(w);
    	result[iter].push_back(h);
    	result[iter].push_back(area);
	}
    return result;
}
int main(int argc, char** argv)
{
  std::vector<vector<int >> v = SingleSegmentMeasure(img);
	for(int iter = 0; iter < v.size(); iter++)
	{	
 	  		int w = v[iter][0];
	  		int h = v[iter][1];
 	  		int area = v[iter][2];
	  		cout << "w: " << w << " h: " << h << " area: " << area << endl;
	}
	auto t3 = system_clock::now();
	auto d1 = duration_cast<microseconds>(t2 - t1);
	auto d2 = duration_cast<microseconds>(t3 - t2);
	printf("read time : %lf s\n", d1.count() / 1000000.);
	printf("cal time : %lf s\n", d2.count() / 1000000.);
  return 0;
}
