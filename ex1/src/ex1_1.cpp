#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

struct Hist {
  uint32_t hist[256];
  uint8_t m;
  int32_t nm;
  uint32_t t;
};

cv::Mat combine_image(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      cv::Mat &img4);
cv::Mat add_salt_and_peper_noise(const cv::Mat &image, uint32_t noise_num);
cv::Mat add_impulse_noise(const cv::Mat &image, uint32_t noise_num,
                          uint8_t threshold = 200);
cv::Mat add_gaussian_noise(const cv::Mat &image, float sigma, float mu = .0f);

struct Hist calc_hist(const cv::Mat image);
cv::Mat mean_filter(const cv::Mat &image, uint8_t kernel_size);
cv::Mat median_filter(const cv::Mat image, uint8_t window_width,
                      uint8_t window_height);

int main() {

  // read image
  const cv::Mat img = cv::imread("../img/source/lena.bmp",
                                 cv::IMREAD_GRAYSCALE);

  // check
  if (img.empty()) {
    std::cout << "Error loading image!" << std::endl;
    return EXIT_FAILURE;
  }

  // // show image
  // cv::imshow("lena", img);
  // cv::waitKey(0);

  // copy image to add text
  cv::Mat src = img.clone();
  cv::putText(src, "source image", cv::Point(50, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_four2one;

  // generate images with salt and peper noise
  cv::Mat img_noise_sp_100 =
      add_salt_and_peper_noise(img, 100); // noise number: 100
  cv::putText(img_noise_sp_100, "noise number: 100", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_sp_500 =
      add_salt_and_peper_noise(img, 500); // noise number: 500
  cv::putText(img_noise_sp_500, "noise number: 500", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_sp_1000 =
      add_salt_and_peper_noise(img, 1000); // noise number: 1000
  cv::putText(img_noise_sp_1000, "noise number: 1000", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  // show images with salt and peper noise
  img_four2one =
      combine_image(src, img_noise_sp_100, img_noise_sp_500, img_noise_sp_1000);
  cv::imshow("Images with salt and peper noise", img_four2one);
  cv::waitKey(0);

  // // save images with salt and peper noise
  // cv::imwrite("../img/generate/lena_sp.jpg", img_four2one);

  // generate images with impulse noise
  cv::Mat img_noise_im_100 = add_impulse_noise(img, 100); // noise number: 100
  cv::putText(img_noise_im_100, "noise number: 100", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_im_500 = add_impulse_noise(img, 500); // noise number: 500
  cv::putText(img_noise_im_500, "noise number: 500", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_im_1000 =
      add_impulse_noise(img, 1000); // noise number: 1000
  cv::putText(img_noise_im_1000, "noise number: 1000", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  // show images with salt and peper noise
  img_four2one =
      combine_image(src, img_noise_im_100, img_noise_im_500, img_noise_im_1000);
  cv::imshow("Images with impulse noise", img_four2one);
  cv::waitKey(0);

  // // save images with impulse noise
  // cv::imwrite("../img/generate/lena_impulse.jpg", img_four2one);

  // generate images with gaussian noise
  cv::Mat img_noise_gs_5 = add_gaussian_noise(img, 1); // noise number: 100
  cv::putText(img_noise_gs_5, "sigma: 5", cv::Point(50, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_gs_10 = add_gaussian_noise(img, 10); // noise number: 500
  cv::putText(img_noise_gs_10, "sigma: 10", cv::Point(50, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::Mat img_noise_gs_50 = add_gaussian_noise(img, 50); // noise number: 1000
  cv::putText(img_noise_gs_50, "sigma: 50", cv::Point(50, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  // show images with gaussian noise
  img_four2one =
      combine_image(src, img_noise_gs_5, img_noise_gs_10, img_noise_gs_50);
  cv::imshow("Images with gaussian noise", img_four2one);
  cv::waitKey(0);

  // // save images with gaussian noise
  // cv::imwrite("../img/generate/lena_gaussian.jpg", img_four2one);

  // generate images with salt and peper noise to filter
  cv::Mat img_noise_sp = add_salt_and_peper_noise(img, 1000);

  cv::Mat img_noise_sp_copy = img_noise_sp.clone();
  cv::putText(img_noise_sp_copy, "noise number: 1000", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  // mean filter
  cv::Mat img_filter_mean_3 = mean_filter(img_noise_sp, 3);
  cv::Mat img_filter_mean_5 = mean_filter(img_noise_sp, 5);

  cv::putText(img_filter_mean_3, "kernel size: 3x3", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::putText(img_filter_mean_5, "kernel size: 5x5", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  img_four2one = combine_image(src, img_noise_sp_copy, img_filter_mean_3,
                               img_filter_mean_5);
  cv::imshow("Mean filter", img_four2one);
  cv::waitKey(0);

  // // save images with mean filter
  // cv::imwrite("../img/generate/lena_mean.jpg", img_four2one);

  // median filter
  cv::Mat img_filter_median_3 = median_filter(img_noise_sp, 3, 3);
  cv::Mat img_filter_median_5 = median_filter(img_noise_sp, 5, 5);

  cv::putText(img_filter_median_3, "kernel size: 3x3", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));
  cv::putText(img_filter_median_5, "kernel size: 5x5", cv::Point(10, 250),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0));

  img_four2one =
      combine_image(src, img_noise_sp_copy, img_filter_median_3, img_filter_median_5);
  cv::imshow("Median filter", img_four2one);
  cv::waitKey(0);

  // // save images with median filter
  // cv::imwrite("../img/generate/lena_median.jpg", img_four2one);

  return 0;
}

/*!
 *  @brief combine four images(same size) to one image
 *  @param img1 image to be combined (upper left)
 *  @param img2 image to be combined (upper right)
 *  @param img3 image to be combined (lower left)
 *  @param img4 image to be combined (lower right)
 */
cv::Mat combine_image(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      cv::Mat &img4) {
  cv::Mat img_out, tmp1, tmp2;
  cv::hconcat(img1, img2, tmp1);
  cv::hconcat(img3, img4, tmp2);
  cv::vconcat(tmp1, tmp2, img_out);
  return img_out;
}

/*!
 *  @brief Add salt and peper noise to a image
 *  @param image Image to be added noise
 *  @param noise_num Number of noise
 */
cv::Mat add_salt_and_peper_noise(const cv::Mat &image, uint32_t noise_num) {
  cv::Mat img_out = image.clone();
  uint8_t flag_white = 1;
  std::srand(std::time(nullptr));
  for (uint32_t i = 0; i < noise_num; i++) {
    uint32_t row_idx = std::rand() % img_out.rows;
    uint32_t col_idx = std::rand() % img_out.cols;
    flag_white = std::rand() % 2;
    uint8_t *p = img_out.ptr<uint8_t>(row_idx);
    if (img_out.type() == CV_8UC1) { // GRAY
      p[col_idx] = flag_white ? 0xff : 0;
    } else if (img_out.type() == CV_8UC3) { // BGR
      if (flag_white) {
        p[col_idx * 3] = 0xff;
        p[col_idx * 3 + 1] = 0xff;
        p[col_idx * 3 + 2] = 0xff;
      } else {
        p[col_idx * 3] = 0;
        p[col_idx * 3 + 1] = 0;
        p[col_idx * 3 + 2] = 0;
      }
    }
  }
  return img_out;
}

/*!
 *  @brief Add impulse noise to a image
 *  @param image image to be added noise
 *  @param noise_num Number of noise
 *  @param threshold Lower bound of white intensity value
 */
cv::Mat add_impulse_noise(const cv::Mat &image, uint32_t noise_num,
                          uint8_t threshold) {
  cv::Mat img_out = image.clone();
  uint8_t flag_white = 1;
  std::srand(std::time(nullptr));
  for (uint32_t i = 0; i < noise_num; i++) {
    uint32_t row_idx = std::rand() % img_out.rows;
    uint32_t col_idx = std::rand() % img_out.cols;
    uint8_t white_point =
        (std::rand() % (0x100 - threshold)) + threshold; // threshold~255
    uint8_t *p = img_out.ptr<uint8_t>(row_idx);
    if (img_out.type() == CV_8UC1) { // GRAY
      p[col_idx] = white_point;
    } else if (img_out.type() == CV_8UC3) { // BGR
      p[col_idx * 3] = white_point;
      p[col_idx * 3 + 1] = white_point;
      p[col_idx * 3 + 2] = white_point;
    }
  }
  return img_out;
}

/*!
 *  @brief Add gaussian noise to a image by cv::RNG
 *  @param img_out image to be added noise
 *  @param sigma standard deviation for gaussian distribution
 *  @param mu mean value for gaussian distribution
 */
cv::Mat add_gaussian_noise_rng(const cv::Mat &image, float sigma, float mu) {
  cv::RNG rng(time(NULL));
  cv::Mat gaussian_noise(image.size(), CV_32FC1);
  rng.fill(gaussian_noise, cv::RNG::NORMAL, mu, sigma);

  cv::Mat img_out;
  cv::add(image, gaussian_noise, img_out, cv::noArray(), CV_8UC1);
  return img_out;
}

/*!
 *  @brief Add gaussian noise to a image
 *  @param img_out image to be added noise
 *  @param sigma standard deviation for gaussian distribution
 *  @param mu mean value for gaussian distribution
 */
cv::Mat add_gaussian_noise(const cv::Mat &image, float sigma, float mu) {
  float r, phi, z1, z2, tmp1, tmp2;
  const float PI = 3.14159;
  cv::Mat img_out;
  cv::Mat gaussian_noise(image.size(), CV_64FC1, cv::Scalar(0));
  std::srand(std::time(nullptr));
  for (uint32_t i = 0; i < image.rows; i++) {
    double *p = gaussian_noise.ptr<double>(i);
    for (uint32_t j = 1; j < image.cols; j += 2) {
      r = (float)(std::rand() % 100) / 100 + 0.01; // 0.01~1
      phi = (float)(std::rand() % 100) / 100;
      tmp1 = 2 * PI * phi;
      tmp2 = sqrt(-2 * log(r));
      z1 = sigma * cos(tmp1) * tmp2;
      z2 = sigma * sin(tmp1) * tmp2;

      p[j - 1] += z1;
      p[j] += z2;
    }
  }

  cv::add(image, gaussian_noise, img_out, cv::noArray(), CV_8UC1);
  return img_out;
}

/*!
 *  @brief Mean Filter
 *  @param image image to filter
 *  @param kernel_size size of kernel (width = height = size)
 */
cv::Mat mean_filter(const cv::Mat &image, uint8_t kernel_size) {
  cv::Mat img_out;
  cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F);
  kernel /= (kernel_size * kernel_size);
  cv::filter2D(image, img_out, image.depth(), kernel, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);
  return img_out;
}

/*!
 *  @brief
 *  @param image image to calculate histogram
 */
struct Hist calc_hist(const cv::Mat image) {
  struct Hist h;
  memset(&h, 0, sizeof(h));
  h.t = image.rows * image.cols / 2;
  for (uint32_t i = 0; i < image.rows; i++) {
    const uint8_t *p = image.ptr<uint8_t>(i);
    for (uint32_t j = 0; j < image.cols; j++) {
      h.hist[p[j]]++;
    }
  }

  for (uint16_t i = 0; i <= 0xff; i++) {
    h.nm += h.hist[i];
    if (h.nm + h.hist[i] >= h.t) {
      h.m = i;
      break;
    }
  }
  return h;
}

/*!
 *  @brief Median Filter
 *  @param image image to filter
 *  @param window_width width of kernel
 *  @param window_height height of kernel
 */
cv::Mat median_filter(const cv::Mat image, uint8_t window_width,
                      uint8_t window_height) {
  cv::Mat img_out = image.clone();
  cv::Mat img_tmp;
  cv::copyMakeBorder(image, img_tmp, window_height / 2, window_height / 2,
                     window_width / 2, window_width / 2, cv::BORDER_REPLICATE);
  for (uint32_t i = 0; i < image.rows; i++) {
    cv::Mat window;
    window = img_tmp(cv::Rect(0, i, window_width, window_height));
    struct Hist h = calc_hist(window);

    uint8_t *p_out = img_out.ptr<uint8_t>(i);
    p_out[0] = h.m;
    uint8_t *p_tmp;

    for (uint32_t j = 1; j < image.cols; j++) {
      for (uint32_t k = i; k < i + window_height; k++) {
        p_tmp = img_tmp.ptr<uint8_t>(k);
        h.hist[p_tmp[j - 1]]--;
        if (p_tmp[j - 1] <= h.m)
          h.nm--;
        h.hist[p_tmp[j + window_width - 1]]++;
        if (p_tmp[j + window_width - 1] <= h.m)
          h.nm++;
      }
      if (h.nm < h.t) {
        do {
          h.m++;
          h.nm += h.hist[h.m];
        } while (h.nm < h.t);
      } else if (h.nm > h.t) {
        while (h.nm > h.t) {
          if (h.nm - h.hist[h.m] < h.t)
            break;
          h.nm -= h.hist[h.m];
          h.m--;
        };
      }
      p_out[j] = h.m;
    }
  }
  return img_out;
}