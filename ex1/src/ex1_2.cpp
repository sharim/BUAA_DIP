#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define COLOR_BGR 0
#define COLOR_GRAY 1
#define COLOR_HSV 2

cv::Mat combine_image(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      cv::Mat &img4, cv::Mat &img5, cv::Mat &img6);
cv::Mat hist2image(const cv::Mat &hist, const uint8_t color_space);
cv::Mat calc_hist(const cv::Mat image, const uint8_t color_space);
cv::Mat calc_hist_c(const cv::Mat hist, const uint8_t color_space);
cv::Mat calc_table(const cv::Mat hist_c, const uint32_t pixel_num,
                   const uint8_t color_space);

int main() {

  cv::Mat img_bgr, img_gray, img_hsv, img_six2one;

  // read image (RGB)
  img_bgr = cv::imread("../img/source/landscape.jpg",
                       cv::IMREAD_COLOR);

  // check
  if (img_bgr.empty()) {
    std::cout << "Error loading image!" << std::endl;
    return EXIT_FAILURE;
  }

  // convert RGB to GRAY
  cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);
  // convert RGB to HSV
  cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);

  // // show source image (GRAY)
  // cv::imshow("GRAY(source)", img_gray);
  // cv::waitKey(0);

  // copy source image to add text
  cv::Mat img_gray_copy = img_gray.clone();
  cv::putText(img_gray_copy, "source image", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255));

  // caclulate histogram of image (GRAY)
  cv::Mat hist_gray_before = calc_hist(img_gray, COLOR_GRAY);
  cv::Mat img_hist_gray_before = hist2image(hist_gray_before, COLOR_GRAY);
  cv::putText(img_hist_gray_before, "histogram(before)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255));

  // caclulate cumulative histogram of image (GRAY)
  cv::Mat hist_c_gray_before = calc_hist_c(hist_gray_before, COLOR_GRAY);
  cv::Mat img_hist_c_gray_before = hist2image(hist_c_gray_before, COLOR_GRAY);
  cv::putText(img_hist_c_gray_before, "cumulative histogram(before)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255));

  // caclulate lookup table
  cv::Mat table_gray =
      calc_table(hist_c_gray_before, img_gray.rows * img_gray.cols, COLOR_GRAY);

  cv::Mat img_gray_e;
  cv::LUT(img_gray, table_gray, img_gray_e);
  cv::putText(img_gray_e, "histogram equalization", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255));

  // caclulate histogram of image (GRAY)
  cv::Mat hist_gray_after = calc_hist(img_gray_e, COLOR_GRAY);
  cv::Mat img_hist_gray_after = hist2image(hist_gray_after, COLOR_GRAY);
  cv::putText(img_hist_gray_after, "histogram(after)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255));
  // caclulate integral histogram of image (GRAY)
  cv::Mat hist_c_gray_after = calc_hist_c(hist_gray_after, COLOR_GRAY);
  cv::Mat img_hist_c_gray_after = hist2image(hist_c_gray_after, COLOR_GRAY);
  cv::putText(img_hist_c_gray_after, "cumulative histogram(after)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255));

  img_six2one = combine_image(img_gray_copy, img_gray_e, img_hist_gray_before,
                              img_hist_gray_after, img_hist_c_gray_before,
                              img_hist_c_gray_after);

  // // save image (GRAY)
  // cv::imwrite("../img/generate/landscape_gray.jpg", img_six2one);

  cv::imshow("Histogram equalization (Gray)", img_six2one);
  cv::waitKey(0);

  // // show source image (RGB)
  // cv::imshow("RGB(source)", img_bgr);
  // cv::waitKey(0);

  // copy source image to add text
  cv::Mat img_bgr_copy = img_bgr.clone();
  cv::putText(img_bgr_copy, "source image", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  // caclulate histogram of image (bgr)
  cv::Mat hist_bgr_before = calc_hist(img_bgr, COLOR_BGR);
  cv::Mat img_hist_bgr_before = hist2image(hist_bgr_before, COLOR_BGR);
  cv::putText(img_hist_bgr_before, "histogram(before)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  // caclulate cumulative histogram of image (bgr)
  cv::Mat hist_c_bgr_before = calc_hist_c(hist_bgr_before, COLOR_BGR);
  cv::Mat img_hist_c_bgr_before = hist2image(hist_c_bgr_before, COLOR_BGR);
  cv::putText(img_hist_c_bgr_before, "cumulative histogram(before)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255, 255, 255));

  // caclulate lookup table
  cv::Mat table_bgr =
      calc_table(hist_c_bgr_before, img_bgr.rows * img_bgr.cols, COLOR_BGR);

  cv::Mat img_bgr_e;
  cv::LUT(img_bgr, table_bgr, img_bgr_e);
  cv::putText(img_bgr_e, "histogram equalization", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  // caclulate histogram of image (BGR)
  cv::Mat hist_bgr_after = calc_hist(img_bgr_e, COLOR_BGR);
  cv::Mat img_hist_bgr_after = hist2image(hist_bgr_after, COLOR_BGR);
  cv::putText(img_hist_bgr_after, "histogram(after)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
  // caclulate cumulative histogram of image (BGR)
  cv::Mat hist_c_bgr_after = calc_hist_c(hist_bgr_after, COLOR_BGR);
  cv::Mat img_hist_c_bgr_after = hist2image(hist_c_bgr_after, COLOR_BGR);
  cv::putText(img_hist_c_bgr_after, "cumulative histogram(after)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255, 255, 255));

  img_six2one = combine_image(img_bgr_copy, img_bgr_e, img_hist_bgr_before,
                              img_hist_bgr_after, img_hist_c_bgr_before,
                              img_hist_c_bgr_after);

  // // save image (RGB)
  // cv::imwrite("../img/generate/landscape_bgr.jpg", img_six2one);

  cv::imshow("Histogram equalization (RGB)", img_six2one);
  cv::waitKey(0);

  // // show source image (HSV)
  cv::cvtColor(img_hsv, img_hsv, cv::COLOR_HSV2BGR);
  // cv::imshow("source image(HSV)", img_hsv);
  // cv::waitKey(0);

  // copy source image to add text
  cv::Mat img_hsv_copy = img_hsv.clone();
  cv::putText(img_hsv_copy, "source image", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  // re-convert RGB to HSV
  cv::cvtColor(img_hsv, img_hsv, cv::COLOR_BGR2HSV);

  // caclulate histogram of image (hsv)
  cv::Mat hist_hsv_before = calc_hist(img_hsv, COLOR_HSV);
  cv::Mat img_hist_hsv_before = hist2image(hist_hsv_before, COLOR_HSV);
  cv::putText(img_hist_hsv_before, "histogram(before)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  // caclulate cumulative histogram of image (hsv)
  cv::Mat hist_c_hsv_before = calc_hist_c(hist_hsv_before, COLOR_HSV);
  cv::Mat img_hist_c_hsv_before = hist2image(hist_c_hsv_before, COLOR_HSV);
  cv::putText(img_hist_c_hsv_before, "cumulative histogram(before)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255, 255, 255));

  // caclulate lookup table
  cv::Mat table_hsv =
      calc_table(hist_c_hsv_before, img_hsv.rows * img_hsv.cols, COLOR_HSV);

  cv::Mat img_hsv_e = img_hsv.clone();
  for (uint32_t i = 0; i < img_hsv.rows; i++) {
    uint8_t *p = img_hsv_e.ptr<uint8_t>(i);
    uint8_t *t = table_hsv.ptr<uint8_t>();
    for (uint32_t j = 0; j < img_hsv_e.cols; j++) {
      p[j * 3 + 2] = t[p[j * 3 + 2]];
    }
  }

  // caclulate histogram of image (HSV)
  cv::Mat hist_hsv_after = calc_hist(img_hsv_e, COLOR_HSV);
  cv::Mat img_hist_hsv_after = hist2image(hist_hsv_after, COLOR_HSV);
  cv::putText(img_hist_hsv_after, "histogram(after)", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
  // caclulate cumulative histogram of image (HSV)
  cv::Mat hist_c_hsv_after = calc_hist_c(hist_hsv_after, COLOR_HSV);
  cv::Mat img_hist_c_hsv_after = hist2image(hist_c_hsv_after, COLOR_HSV);
  cv::putText(img_hist_c_hsv_after, "cumulative histogram(after)",
              cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
              cv::Scalar(255, 255, 255));

  cv::cvtColor(img_hsv_e, img_hsv_e, cv::COLOR_HSV2BGR);
  cv::putText(img_hsv_e, "histogram equalization", cv::Point(20, 20),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  img_six2one = combine_image(img_hsv_copy, img_hsv_e, img_hist_hsv_before,
                              img_hist_hsv_after, img_hist_c_hsv_before,
                              img_hist_c_hsv_after);

  // // save image (RGB -> HSV -> RGB)
  // cv::imwrite("../img/generate/landscape_hsv.jpg", img_six2one);

  cv::imshow("Histogram equalization (HSV)", img_six2one);
  cv::waitKey(0);

  return 0;
}

cv::Mat combine_image(cv::Mat &img1, cv::Mat &img2, cv::Mat &img3,
                      cv::Mat &img4, cv::Mat &img5, cv::Mat &img6) {
  cv::Mat img_out, tmp1, tmp2, tmp3;
  cv::hconcat(img1, img2, tmp1);
  cv::hconcat(img3, img4, tmp2);
  cv::hconcat(img5, img6, tmp3);
  cv::vconcat(tmp1, tmp2, img_out);
  cv::vconcat(img_out, tmp3, img_out);
  return img_out;
}

cv::Mat hist2image(const cv::Mat &hist, const uint8_t color_space) {
  int hist_w = 500, hist_h = 313, histSize = 256;
  int bin_w = cvRound((double)hist_w / histSize);
  cv::Mat histImage;

  if (color_space == COLOR_GRAY) {
    histImage = cv::Mat(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
  } else if (color_space == COLOR_BGR || color_space == COLOR_HSV) {
    histImage = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
  }

  for (int i = 1; i < histSize; i++) {
    if (color_space == COLOR_GRAY) {
      // std::cout << cvRound((double)h[i - 1] / hist_h) << std::endl;
      const double *h = hist.ptr<double>();
      cv::line(histImage,
               cv::Point(bin_w * (i - 1),
                         hist_h - cvRound(h[i - 1] / h[257] * hist_h)),
               cv::Point(bin_w * (i), hist_h - cvRound(h[i] / h[257] * hist_h)),
               cv::Scalar(255), 2, 8, 0);
    } else if (color_space == COLOR_BGR) {
      const cv::Vec3d *h = hist.ptr<cv::Vec3d>();
      cv::line(histImage,
               cv::Point(bin_w * (i - 1),
                         hist_h - cvRound(h[i - 1][0] / h[257][0] * hist_h)),
               cv::Point(bin_w * (i),
                         hist_h - cvRound(h[i][0] / h[257][0] * hist_h)),
               cv::Scalar(255, 0, 0), 2, 8, 0);
      cv::line(histImage,
               cv::Point(bin_w * (i - 1),
                         hist_h - cvRound(h[i - 1][1] / h[257][1] * hist_h)),
               cv::Point(bin_w * (i),
                         hist_h - cvRound(h[i][1] / h[257][1] * hist_h)),
               cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::line(histImage,
               cv::Point(bin_w * (i - 1),
                         hist_h - cvRound(h[i - 1][2] / h[257][2] * hist_h)),
               cv::Point(bin_w * (i),
                         hist_h - cvRound(h[i][2] / h[257][2] * hist_h)),
               cv::Scalar(0, 0, 255), 2, 8, 0);
    } else if (color_space == COLOR_HSV) {
      const double *h = hist.ptr<double>();
      cv::line(
          histImage,
          cv::Point(bin_w * (i - 1),
                    hist_h - cvRound(h[i - 1] / h[257] * hist_h)),
          cv::Point(bin_w * (i), hist_h - cvRound(h[i] / h[257] * hist_h)),
          cv::Scalar(255, 255, 255), 2, 8, 0);
    }
  }

  return histImage;
}

cv::Mat calc_hist(const cv::Mat image, const uint8_t color_space) {
  cv::Mat hist;
  int max[3] = {0, 0, 0};
  if (color_space == COLOR_GRAY) {
    hist = cv::Mat(1, 258, CV_64FC1, cv::Scalar(0));
    double *h = hist.ptr<double>();
    for (uint32_t i = 0; i < image.rows; i++) {
      const uint8_t *p = image.ptr<uint8_t>(i);
      for (uint32_t j = 0; j < image.cols; j++) {
        h[p[j]]++;
        if (h[p[j]] > max[0]) {
          max[0] = h[p[j]];
        }
      }
    }
    h[257] = max[0];
    for (uint32_t i = 0; i < image.rows; i++) {
      if (h[i]) {
        h[256] = i;
        break;
      }
    }
    return hist;
  } else if (color_space == COLOR_HSV) {
    hist = cv::Mat(1, 258, CV_64FC1, cv::Scalar(0));
    double *h = hist.ptr<double>();
    for (uint32_t i = 0; i < image.rows; i++) {
      const uint8_t *p = image.ptr<uint8_t>(i);
      for (uint32_t j = 0; j < image.cols; j++) {
        h[p[j * 3 + 2]]++;
        if (h[p[j * 3 + 2]] > max[0]) {
          max[0] = h[p[j * 3 + 2]];
        }
      }
    }
    h[257] = max[0];
    for (uint32_t i = 0; i < image.rows; i++) {
      if (h[i]) {
        h[256] = i;
        break;
      }
    }
    return hist;
  } else if (color_space == COLOR_BGR) {
    hist = cv::Mat(1, 258, CV_64FC3, cv::Scalar(0));
    cv::Vec3d *h = hist.ptr<cv::Vec3d>();
    for (uint32_t i = 0; i < image.rows; i++) {
      const cv::Vec3b *p = image.ptr<cv::Vec3b>(i);
      for (uint32_t j = 0; j < image.cols; j++) {
        h[p[j][0]][0]++;
        h[p[j][1]][1]++;
        h[p[j][2]][2]++;
        if (h[p[j][0]][0] > max[0])
          max[0] = h[p[j][0]][0];
        if (h[p[j][1]][1] > max[1])
          max[1] = h[p[j][1]][1];
        if (h[p[j][2]][2] > max[2])
          max[2] = h[p[j][2]][2];
      }
    }
    h[257][0] = max[0];
    h[257][1] = max[1];
    h[257][2] = max[2];
    uint8_t flag_break = 0;
    for (uint32_t i = 0; i < 256; i++) {
      if (h[i][0] && ((flag_break & 0b1) == 0)) {
        h[256][0] = i;
        flag_break |= 0b1;
      }
      if (h[i][1] && ((flag_break & 0b10) == 0)) {
        h[256][1] = i;
        flag_break |= 0b10;
      }
      if (h[i][2] && ((flag_break & 0b100) == 0)) {
        h[256][2] = i;
        flag_break |= 0b100;
      }
      if (flag_break == 0b111)
        break;
    }
    return hist;
  }
}

cv::Mat calc_hist_c(const cv::Mat hist, const uint8_t color_space) {
  cv::Mat hist_c;
  if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
    hist_c = cv::Mat(1, 258, CV_64FC1, cv::Scalar(0));
  } else if (color_space == COLOR_BGR) {
    hist_c = cv::Mat(1, 258, CV_64FC3, cv::Scalar(0, 0, 0));
  }
  double *c = hist_c.ptr<double>();
  const double *h = hist.ptr<double>();

  for (uint16_t i = 0; i <= 0xff; i++) {
    if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
      c[i] = i ? (h[i] + c[i - 1]) : 0;
    } else if (color_space == COLOR_BGR) {
      c[i * 3] = i ? (h[i * 3] + c[i * 3 - 3]) : 0;
      c[i * 3 + 1] = i ? (h[i * 3 + 1] + c[i * 3 - 2]) : 0;
      c[i * 3 + 2] = i ? (h[i * 3 + 2] + c[i * 3 - 1]) : 0;
    }
  }

  if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
    c[256] = c[cvRound(h[256])];
  } else if (color_space == COLOR_BGR) {
    c[256 * 3 + 0] = c[cvRound(h[256 * 3 + 0])];
    c[256 * 3 + 1] = c[cvRound(h[256 * 3 + 1])];
    c[256 * 3 + 2] = c[cvRound(h[256 * 3 + 2])];
  }

  if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
    c[257] = c[255];
  } else if (color_space == COLOR_BGR) {
    c[257 * 3 + 0] = c[255 * 3 + 0];
    c[257 * 3 + 1] = c[255 * 3 + 1];
    c[257 * 3 + 2] = c[255 * 3 + 2];
  }

  return hist_c;
}

cv::Mat calc_table(cv::Mat hist_c, const uint32_t pixel_num,
                   const uint8_t color_space) {
  cv::Mat table;
  if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
    table.create(1, 256, CV_8UC1);
  } else if (color_space == COLOR_BGR) {
    table.create(1, 256, CV_8UC3);
  }
  uint8_t *t = table.ptr<uint8_t>();
  const double*c = hist_c.ptr<double>();

  for (uint16_t i = 0; i <= 0xff; i++) {
    if (color_space == COLOR_GRAY || color_space == COLOR_HSV) {
      t[i] = cvRound((double)(c[i] - c[256]) / (pixel_num - c[256]) * 0xff);
    } else if (color_space == COLOR_BGR) {
      t[i * 3 + 0] = cvRound((c[i * 3 + 0] - c[256 * 3 + 0]) /
                             (pixel_num - c[256 * 3 + 0]) * 0xff);
      t[i * 3 + 1] = cvRound((c[i * 3 + 1] - c[256 * 3 + 1]) /
                             (pixel_num - c[256 * 3 + 1]) * 0xff);
      t[i * 3 + 2] = cvRound((c[i * 3 + 2] - c[256 * 3 + 2]) /
                             (pixel_num - c[256 * 3 + 2]) * 0xff);
    }
  }

  return table;
}
