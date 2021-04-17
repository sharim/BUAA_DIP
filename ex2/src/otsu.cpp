#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  cv::Mat img = cv::imread("/home/sharim/MyProjects/DIP/ex2/img/cells.bmp",
                           cv::IMREAD_GRAYSCALE);

  cv::Mat hist;
  const int channels = 0;
  const int histSize = 10;
  float range[] = {0, 256};
  const float *ranges = {range};
  cv::calcHist(&img, 1, &channels, cv::Mat(), hist, 1, &histSize, &ranges, true, false);
  std::cout << hist << std::endl;

  return 0;
}