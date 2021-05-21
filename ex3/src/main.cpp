#include "ex3.hpp"
#include <opencv2/opencv.hpp>

// #define SAVE_IMG
// #define TASK_1
// #define TASK_2
#define TASK_3
// #define TASK_4

int main(int argc, char **argv) {

  /* ---------- Task 1 ---------- */
#ifdef TASK_1
  cv::Mat img_rect =
      cv::imread("../img/src_img/rect.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_45 =
      cv::imread("../img/src_img/rect-45度.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_move =
      cv::imread("../img/src_img/rect-move.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_rect_dft, img_rect_45_dft, img_rect_move_dft;

  ex3::freq(img_rect, img_rect_dft, true, true, true);
  ex3::freq(img_rect_45, img_rect_45_dft, true, true, true);
  ex3::freq(img_rect_move, img_rect_move_dft, true, true, true);

  cv::imshow("rect", img_rect_dft);
  cv::imshow("rect45", img_rect_45_dft);
  cv::imshow("rctmove", img_rect_move_dft);
  cv::waitKey(0);

#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/rect_dft.jpg", img_rect_dft);
  cv::imwrite("../img/gen_img/rect_45_dft.jpg", img_rect_45_dft);
  cv::imwrite("../img/gen_img/rect_move.jpg", img_rect_move_dft);
#endif
#endif

  /* ---------- Task 2 ---------- */
#ifdef TASK_2
  cv::Mat img_grid =
      cv::imread("../img/src_img/grid.bmp", cv::IMREAD_GRAYSCALE);
  cv::Mat img_grid_dft;

  ex3::freq(img_grid, img_grid_dft, true, true, true);
  cv::imshow("grid_dft", img_grid_dft);
  cv::waitKey(0);

  cv::Mat img_grid_low, img_grid_high, img_grid_low_dft, img_grid_high_dft;

  ex3::freq_fliter(img_grid, img_grid_low, ex3::IDEAL_LOWPASS, 10);
  cv::imshow("grid_low", img_grid_low);
  ex3::freq(img_grid_low, img_grid_low_dft, true, true, true);
  cv::imshow("grid_low_dft", img_grid_low_dft);

  ex3::freq_fliter(img_grid, img_grid_high, ex3::IDEAL_HIGHPASS, 10);
  cv::imshow("grid_high", img_grid_high);
  ex3::freq(img_grid_high, img_grid_high_dft, true, true, true);
  cv::imshow("grid_high_dft", img_grid_high_dft);
  cv::waitKey(0);

#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/grid_low.jpg", img_grid_low);
  cv::imwrite("../img/gen_img/grid_low_dft.jpg", img_grid_low_dft);
  cv::imwrite("../img/gen_img/grid_high.jpg", img_grid_high);
  cv::imwrite("../img/gen_img/grid_high_dft.jpg", img_grid_high_dft);
#endif
#endif

  /* ---------- Task 3 ---------- */
#ifdef TASK_3
  cv::Mat img_lena =
      cv::imread("../img/src_img/lena.bmp", cv::IMREAD_GRAYSCALE);
  cv::imshow("lena", img_lena);

  cv::Mat img_lena_gaussian_fliter, img_lena_gaussian_fliter_dft;
  ex3::freq_fliter(img_lena, img_lena_gaussian_fliter, ex3::GAUSSIAN, 15);
  cv::imshow("lena_gaussian_fliter", img_lena_gaussian_fliter);
  ex3::freq(img_lena_gaussian_fliter, img_lena_gaussian_fliter_dft, true, true,
            true);
  cv::imshow("lena_gaussian_fliter_dft", img_lena_gaussian_fliter_dft);

  cv::Mat img_lena_butterworth_fliter, img_lena_butterworth_fliter_dft;
  ex3::freq_fliter(img_lena, img_lena_butterworth_fliter, ex3::BUTTERWORTH, 15);
  cv::imshow("lena_butterworth_fliter", img_lena_butterworth_fliter);
  ex3::freq(img_lena_butterworth_fliter, img_lena_butterworth_fliter_dft, true,
            true, true);
  cv::imshow("lena_butterworth_fliter_dft", img_lena_butterworth_fliter_dft);
  cv::waitKey(0);

#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/lena_gaussian_fliter.jpg", img_lena_gaussian_fliter);
  cv::imwrite("../img/gen_img/lena_gaussian_fliter_dft.jpg",
              img_lena_gaussian_fliter_dft);
  cv::imwrite("../img/gen_img/lena_butterworth_fliter.jpg",
              img_lena_butterworth_fliter);
  cv::imwrite("../img/gen_img/lena_butterworth_fliter_dft.jpg",
              img_lena_butterworth_fliter_dft);
#endif
#endif

  /* ---------- Task 4 ---------- */
#ifdef TASK_4
  cv::Mat img_cave =
      cv::imread("../img/src_img/cave.jpg", cv::IMREAD_GRAYSCALE);
  cv::imshow("cave", img_cave);

  cv::Mat img_cave_dft, img_cave_homo_fliter, img_cave_homo_dft;

  ex3::freq(img_cave, img_cave_dft, true, true, true);
  cv::imshow("cave_dft", img_cave_dft);
  ex3::homomorphic_fliter(img_cave, img_cave_homo_fliter, 5, 10, 0.001, 5);
  cv::imshow("cave_homo", img_cave_homo_fliter);
  ex3::freq(img_cave_homo_fliter, img_cave_homo_dft, true, true, true);
  cv::imshow("cave_homo_dft", img_cave_homo_dft);
  cv::waitKey(0);

#ifdef SAVE_IMG
  cv::imwrite("../img/gen_img/cave_dft.jpg",
              img_cave_dft);
  cv::imwrite("../img/gen_img/cave_homo.jpg",
              img_cave_homo_fliter);
  cv::imwrite("../img/gen_img/cave_homo_dft.jpg", img_cave_homo_dft);
#endif
#endif

  return 0;
}