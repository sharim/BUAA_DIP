#ifndef EX2_HPP
#define EX2_HPP

#include <opencv2/opencv.hpp>

namespace ex2 {

enum Kernels {
  SOBEL = 0x10,
  ROBERTS = 0x20,
  LAPLACE_4 = 0x30,
  LAPLACE_8 = 0x31,
  PREWITT = 0x40
};

const float Pi = 3.1415927f;

/*!
 *  @brief Rotate a kernel 45 degrees clockwise
 *  @param Kernel_in kernel to be rotated
 */
cv::Mat kernel_rotate_45(cv::Mat kernel_in);
/*!
 *  @brief Get edges of image
 *  @param img image to get edges
 *  @param kernel_select choose which kernel to use, default is SOBEL
 *  @param en_binarization enable to binarize the edges-image, default is true
 *  @param threshold threshold to binarize the edges-image, default is
 * -1(auto-choose the threshold by OTSU)
 */
cv::Mat edge(cv::Mat img, uchar kernel_select = SOBEL,
             bool en_binarization = true, int threshold = -1);
/*!
 *  @brief calculate a threshold for binarizing images by otsu algorithm
 *  @param img image for calculating it's threshold
 */
uchar otsu(cv::Mat img);
/*!
 *  @brief binarize image with threshold
 *  @param src source image
 *  @param dst destination image
 *  @param threshold threshold to be used to binarize the image
 *  @param bk_black default is true, if bk_black is true and the value of pixel
 * is greater than the threshold, set it to 0xff, otherwise 0;
 */
void binarize(cv::InputArray src, cv::OutputArray dst, uchar threshold,
              bool bk_black = true);
/*!
 *  @brief edge detection by canny algorithm
 *  @param img input image
 *  @param edges output image
 *  @param low_threshold low threshold for the hysteresis procedure
 *  @param high_threshold low threshold for the hysteresis procedure
 *  @param sigma gaussian kernel standard deviation, default is -1(disable
 * gaussian filter)
 *  @param gaussian_size gaussian kernel size, default is 0(auto-choose size of
 * gaussian kernel)
 */
void canny(cv::InputArray img, cv::OutputArray edges, uchar low_threshold,
           uchar high_threshold, float sigma = -1, uchar gaussian_size = 0);

cv::Mat kernel_rotate_45(cv::Mat kernel_in) {
  cv::Mat kernel_out = kernel_in.clone();
  char *p_in, *p_in_up, *p_in_down, *p_out;
  for (int i = 0; i < kernel_in.rows / 2; i++) {
    for (int j = i; j < kernel_in.rows - i; j++) {
      p_in = kernel_in.ptr<char>(j);
      p_out = kernel_out.ptr<char>(j);
      if (j)
        p_in_up = kernel_in.ptr<char>(j - 1);
      if (j != kernel_in.rows - 1)
        p_in_down = kernel_in.ptr<char>(j + 1);
      for (int k = i; k < kernel_in.cols - i;) {
        if (j == i) {
          if (k == i)
            p_out[k] = p_in_down[k];
          else
            p_out[k] = p_in[k - 1];
          k++;
        } else if (j == kernel_in.rows - i - 1) {
          if (k == kernel_in.cols - i - 1)
            p_out[k] = p_in_up[k];
          else
            p_out[k] = p_in[k + 1];
          k++;
        } else if (k == i) {
          p_out[k] = p_in_down[k];
          k = kernel_in.cols - i - 1;
        } else {
          p_out[k] = p_in_up[k];
          k++;
        }
      }
    }
  }
  return kernel_out;
}

cv::Mat edge(cv::Mat img, uchar kernel_select, bool en_binarization,
             int threshold) {
  CV_Assert(img.type() == CV_8UC1);
  cv::Mat kernel;
  cv::Mat img_out = cv::Mat::zeros(img.size(), img.type());
  cv::Mat img_grad;
  uchar kernel_num;
  switch (kernel_select) {
  case SOBEL:
    kernel = (cv::Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    kernel_num = 8;
    break;
  case ROBERTS:
    kernel = (cv::Mat_<char>(2, 2) << 1, 0, 0, -1);
    kernel_num = 2;
    break;
  case LAPLACE_4:
    kernel = (cv::Mat_<char>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    kernel_num = 1;
    break;
  case LAPLACE_8:
    kernel = (cv::Mat_<char>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    kernel_num = 1;
    break;
  case PREWITT:
    kernel = (cv::Mat_<char>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    kernel_num = 8;
    break;
  default:
    std::cout << "error" << std::endl;
    break;
  }
  for (uchar i = 0; i < kernel_num; i++) {
    cv::filter2D(img, img_grad, img.depth(), kernel, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    for (int j = 0; j < img_out.rows; j++) {
      uchar *p = img_out.ptr<uchar>(j);
      uchar *q = img_grad.ptr<uchar>(j);
      for (int k = 0; k < img_out.cols; k++) {
        q[k] = q[k] >= 0 ? q[k] : -q[k];
        p[k] = p[k] >= q[k] ? p[k] : q[k];
      }
    }
    kernel = kernel_rotate_45(kernel);
  }

  if (en_binarization) {
    if (threshold == -1)
      threshold = otsu(img_out);
    binarize(img_out, img_out, threshold);
  }

  return img_out;
}

uchar otsu(cv::Mat img) {
  CV_Assert(img.type() == CV_8UC1);
  cv::Mat hist;
  const int channels = 0;
  const int histSize = 256;
  float range[] = {0, 256};
  const float *ranges = {range};
  cv::calcHist(&img, 1, &channels, cv::Mat(), hist, 1, &histSize, &ranges, true,
               false);
  hist /= (img.rows * img.cols);

  float w0 = 0, w1;
  float u = 0, ut = 0, u0, u1;
  float s, s_max = 0;
  uchar s_idx;
  float *p = hist.ptr<float>();
  for (int i = 0; i < 256; i++) {
    u += i * p[i];
  }
  for (int i = 0; i < 256; i++) {
    w0 += p[i];
    w1 = 1 - w0;
    ut += i * p[i];
    u0 = ut / w0;
    u1 = (u - ut) / w1;
    s = w0 * w1 * (u1 - u0) * (u1 - u0);
    if (s > s_max) {
      s_max = s;
      s_idx = i;
    }
  }
  return s_idx;
}

void binarize(cv::InputArray src, cv::OutputArray dst, uchar threshold,
              bool bk_black) {
  cv::Mat img_in = src.getMat();
  CV_Assert(img_in.type() == CV_8UC1);
  dst.create(img_in.size(), img_in.type());
  cv::Mat img_out = dst.getMat();
  for (int i = 0; i < img_in.rows; i++) {
    uchar *p = img_in.ptr<uchar>(i);
    uchar *q = img_out.ptr<uchar>(i);
    for (int j = 0; j < img_in.cols; j++)
      if (bk_black)
        q[j] = p[j] >= threshold ? 0xff : 0;
      else
        q[j] = p[j] >= threshold ? 0 : 0xff;
  }
}

void canny(cv::InputArray img, cv::OutputArray edges, uchar low_threshold,
           uchar high_threshold, float sigma, uchar gaussian_size) {

  cv::Mat _img = img.getMat();
  CV_Assert(_img.type() == CV_8UC1);
  edges.create(_img.size(), _img.type());
  cv::Mat _edges = edges.getMat();

  // - Step 1: Smoothing image with a Gaussian filter
  cv::Mat img_blur;
  if (sigma > 0) {
    if (!gaussian_size)
      gaussian_size = cvRound(sigma * 6) | 1;
    cv::GaussianBlur(_img, img_blur, cv::Size(gaussian_size, gaussian_size),
                     sigma);
  } else {
    img_blur = _img;
  }

  // - Step 2: Finding Intensity Gradient of the Image
  cv::Mat sobel_x = (cv::Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  cv::Mat sobel_y = (cv::Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
  cv::Mat edge_x, edge_y, edge(img_blur.size(), CV_8UC2);
  cv::filter2D(img_blur, edge_x, CV_32F, sobel_x);
  cv::filter2D(img_blur, edge_y, CV_32F, sobel_y);
  float angle;
  for (int i = 0; i < img_blur.rows; i++) {
    cv::Vec2b *p = edge.ptr<cv::Vec2b>(i);
    float *ex = edge_x.ptr<float>(i);
    float *ey = edge_y.ptr<float>(i);
    for (int j = 0; j < img_blur.cols; j++) {
      p[j][0] = cvRound(sqrt(ex[j] * ex[j] + ey[j] * ey[j]));
      angle = (float)ey[j] / (float)ex[j];
      if ((angle > -0.4142) && (angle <= 0.4142))
        p[j][1] = 3;
      else if ((angle > 0.4142) && (angle <= 2.4142))
        p[j][1] = 0;
      else if ((angle > 2.4142) || (angle < -2.4142))
        p[j][1] = 1;
      else
        p[j][1] = 2;
    }
  }

  // - Step 3: Non-maximum Suppression
  cv::copyMakeBorder(edge, edge, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0));
  cv::Mat thin_edge(_edges.size(), CV_8UC1);
  for (int i = 1; i < edge.rows - 1; i++) {
    cv::Vec2b *p_u = edge.ptr<cv::Vec2b>(i - 1);
    cv::Vec2b *p_c = edge.ptr<cv::Vec2b>(i);
    cv::Vec2b *p_d = edge.ptr<cv::Vec2b>(i + 1);
    uchar *q = thin_edge.ptr<uchar>(i - 1);
    for (int j = 1; j < edge.cols - 1; j++) {
      switch (p_c[j][1]) {
      case 0:
        q[j - 1] =
            ((p_c[j][0] >= p_u[j + 1][0]) && (p_c[j][0] >= p_d[j - 1][0]))
                ? p_c[j][0]
                : 0;
        break;
      case 1:
        q[j - 1] = ((p_c[j][0] >= p_u[j][0]) && (p_c[j][0] >= p_d[j][0]))
                       ? p_c[j][0]
                       : 0;
        break;
      case 2:
        q[j - 1] =
            ((p_c[j][0] >= p_u[j - 1][0]) && (p_c[j][0] >= p_d[j + 1][0]))
                ? p_c[j][0]
                : 0;
        break;
      case 3:
        q[j - 1] =
            ((p_c[j][0] >= p_c[j + 1][0]) && (p_c[j][0] >= p_c[j - 1][0]))
                ? p_c[j][0]
                : 0;
        break;
      default:
        std::cout << "error" << std::endl;
        break;
      }
    }
  }

  // - Step 4: Hysteresis Thresholding
  cv::copyMakeBorder(thin_edge, thin_edge, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  for (int i = 1; i < thin_edge.rows - 1; i++) {
    uchar *p_u = thin_edge.ptr<uchar>(i - 1);
    uchar *p_c = thin_edge.ptr<uchar>(i);
    uchar *p_d = thin_edge.ptr<uchar>(i + 1);
    uchar *q = _edges.ptr<uchar>(i - 1);
    for (int j = 1; j < thin_edge.cols - 1; j++) {
      if (p_c[j] <= low_threshold)
        q[j - 1] = 0;
      else if (p_c[j] >= high_threshold)
        q[j - 1] = 0xff;
      else {
        uchar *p;
        bool isEdge = false;
        for (int k = 0; k < 9; k++) {
          if (k < 3)
            p = p_u;
          else if (k < 6)
            p = p_c;
          else
            p = p_d;
          if (p[j - 1 + k % 3] >= high_threshold) {
            isEdge = true;
            break;
          }
        }
        q[j - 1] = isEdge ? 0xff : 0;
      }
    }
  }
}

} // namespace ex2

#endif