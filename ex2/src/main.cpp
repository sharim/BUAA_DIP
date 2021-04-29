#include "ex2.hpp"
#include <opencv2/opencv.hpp>

// #define SAVE_IMG

int main(int argc, char **argv) {

  /* ---------- Read images ---------- */
  cv::Mat img_cells =
      cv::imread("../img/src_img/cells.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_lena =
      cv::imread("../img/src_img/lena.bmp", cv::IMREAD_GRAYSCALE);

  /* ---------- task 1: Thresholding image by OTSU ---------- */
  uchar threshold = ex2::otsu(img_cells);
  std::cout << "Best threshold: " << (int)threshold << std::endl;
  cv::Mat img_cells_binary;
  ex2::binarize(img_cells, img_cells_binary, threshold);
  cv::imshow("Thresholding by OTSU", img_cells_binary);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_cells_binary.jpg", img_cells_binary);
#endif

  /* ---------- task 2: Edge Detection ---------- */
  // No thresholding
  cv::Mat img_lena_edge_sobel = ex2::edge(img_lena, ex2::SOBEL, false);
  cv::imshow("Sobel", img_lena_edge_sobel);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_sobel.jpg", img_lena_edge_sobel);
#endif
  cv::Mat img_lena_edge_roberts = ex2::edge(img_lena, ex2::ROBERTS, false);
  cv::imshow("Roberts", img_lena_edge_roberts);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_roberts.jpg",
              img_lena_edge_roberts);
#endif
  cv::Mat img_lena_edge_laplace4 = ex2::edge(img_lena, ex2::LAPLACE_4, false);
  cv::imshow("Laplace4", img_lena_edge_laplace4);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_laplace4.jpg",
              img_lena_edge_laplace4);
#endif
  cv::Mat img_lena_edge_laplace8 = ex2::edge(img_lena, ex2::LAPLACE_8, false);
  cv::imshow("Laplace8", img_lena_edge_laplace8);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_laplace8.jpg",
              img_lena_edge_laplace8);
#endif
  cv::Mat img_lena_edge_prewitt = ex2::edge(img_lena, ex2::PREWITT, false);
  cv::imshow("Prewitt", img_lena_edge_prewitt);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_prewitt.jpg",
              img_lena_edge_prewitt);
#endif
  // Thresholding
  img_lena_edge_sobel = ex2::edge(img_lena, ex2::SOBEL);
  cv::imshow("Sobel(thresholding)", img_lena_edge_sobel);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_sobel_thresholding.jpg",
              img_lena_edge_sobel);
#endif
  img_lena_edge_roberts = ex2::edge(img_lena, ex2::ROBERTS);
  cv::imshow("Roberts(thresholding)", img_lena_edge_roberts);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_roberts_thresholding.jpg",
              img_lena_edge_roberts);
#endif
  img_lena_edge_laplace4 = ex2::edge(img_lena, ex2::LAPLACE_4);
  cv::imshow("Laplace4(thresholding)", img_lena_edge_laplace4);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_laplace4_thresholding.jpg",
              img_lena_edge_laplace4);
#endif
  img_lena_edge_laplace8 = ex2::edge(img_lena, ex2::LAPLACE_8);
  cv::imshow("Laplace8(thresholding)", img_lena_edge_laplace8);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_laplace8_thresholding.jpg",
              img_lena_edge_laplace8);
#endif
  img_lena_edge_prewitt = ex2::edge(img_lena, ex2::PREWITT);
  cv::imshow("Prewitt(thresholding)", img_lena_edge_prewitt);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_prewitt_thresholding.jpg",
              img_lena_edge_prewitt);
#endif

  /* ---------- task 2: Edge Detection(by Canny) ---------- */
  cv::Mat img_lena_edge_canny;
  ex2::canny(img_lena, img_lena_edge_canny, 50, 100, 1);
  cv::imshow("Canny", img_lena_edge_canny);
  cv::waitKey(0);
#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/img_lena_edge_canny.jpg",
              img_lena_edge_canny);
#endif

  return 0;
}
