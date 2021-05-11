#include "ex3.hpp"
#include <opencv2/opencv.hpp>

// #define SAVE_IMG
// #define TASK_1
#define TASK_2
// #define TASK_3
// #define TASK_4

int main(int argc, char **argv) {

  /* ---------- Read images ---------- */
  cv::Mat img_cave =
      cv::imread("../img/src_img/cave.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat img_lena =
      cv::imread("../img/src_img/lena.bmp", cv::IMREAD_GRAYSCALE);

  /* ---------- Task 1 ---------- */
#ifdef TASK_1
  cv::Mat img_rect =
      cv::imread("../img/src_img/rect.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_45 =
      cv::imread("../img/src_img/rect-45度.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_move =
      cv::imread("../img/src_img/rect-move.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_dft, img_rect_45_dft, img_rect_move_dft;

  ex3::freq(img_rect, img_rect_dft, true, true);
  ex3::freq(img_rect_45, img_rect_45_dft, true, true);
  ex3::freq(img_rect_move, img_rect_move_dft, true, true);

  cv::imshow("rect", img_rect_dft);
  cv::imshow("rect45", img_rect_45_dft);
  cv::imshow("rctmove", img_rect_move_dft);
  cv::waitKey(0);

#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/rect_dft.jpg", img_rect_dft);
  cv::imwrite("../img/gen_img/rect_45_dft.jpg", img_rect_45_dft);
  cv::imwrite("../img/gen_img/img_rect_move.jpg", img_rect_move_dft);
#endif
#endif

  /* ---------- Task 2 ---------- */
#ifdef TASK_2
  cv::Mat img_grid =
      cv::imread("../img/src_img/grid.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_grid_dft;

  ex3::freq(img_grid, img_grid_dft, true, true);
  cv::imshow("grid_dft", img_grid_dft);
  cv::waitKey(0);
#endif

  /* ---------- Task 3 ---------- */
#ifdef TASK_3
#endif

  /* ---------- Task 4 ---------- */
#ifdef TASK_4
#endif

  return 0;
}