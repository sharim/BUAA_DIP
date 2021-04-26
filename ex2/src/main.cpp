#include <opencv2/opencv.hpp>
#include "ex2.hpp"

int main(int argc, char **argv) {
  // cv::Mat img_cells = cv::imread("../img/cells.bmp", cv::IMREAD_GRAYSCALE);
  // uchar t = ex2::otsu(img_cells);

  // std::cout << "t:" << (int)t << ' ' << atan(1) << std::endl;

  // ex2::binarize(img_cells, img_cells, t);
  // cv::imshow("1", img_cells);
  // cv::waitKey(0);

  cv::Mat img = cv::imread("../img/lena.bmp", cv::IMREAD_GRAYSCALE);
  // img = ex2::median_filter(img, 3, 3);
  cv::medianBlur(img, img, 3);
  cv::imshow("SRC", img);
  cv::waitKey(0);
  cv::Mat img_o = ex2::edge(img);
  // t = ex2::otsu(img_o);
  // ex2::binarize(img_o, img_o, t);
  cv::imshow("sobel", img_o);
  cv::waitKey(0);

  // img_o = ex2::edge(img, ex2::ROBERTS);
  // // t = ex2::otsu(img_o);
  // // ex2::binarize(img_o, img_o, t);
  // cv::imshow("ROBERTS", img_o);
  // cv::waitKey(0);

  // img_o = ex2::edge(img, ex2::LAPLACE_4);
  // // t = ex2::otsu(img_o);
  // // ex2::binarize(img_o, img_o, t);
  // cv::imshow("LAPLACE_4", img_o);
  // cv::waitKey(0);

  // img_o = ex2::edge(img, ex2::LAPLACE_8);
  // // t = ex2::otsu(img_o);
  // // ex2::binarize(img_o, img_o, t);
  // cv::imshow("LAPLACE_8", img_o);
  // cv::waitKey(0);

  // img_o = ex2::edge(img, ex2::PREWITT);
  // // t = ex2::otsu(img_o);
  // // ex2::binarize(img_o, img_o, t);
  // cv::imshow("PREWITT", img_o);
  // cv::waitKey(0);

  ex2::canny(img, img_o, 50, 100, 3, 1);

  imshow("canny", img_o);
  cv::waitKey(0);

  return 0;
}
