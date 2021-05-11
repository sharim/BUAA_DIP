#ifndef EX3_HPP
#define EX3_HPP

#include <opencv2/opencv.hpp>

namespace ex3 {
void circular_shift(cv::InputArray src, cv::OutputArray dst, cv::Point delta);
void fft_shift(cv::InputArray src, cv::OutputArray dst);
void ifft_shift(cv::InputArray src, cv::OutputArray dst);
void FT(cv::InputArray src, cv::OutputArray dst, bool _mag = false,
        bool log = false);
void freq(cv::InputArray src, cv::OutputArray dst, bool shift = false,
          bool _mag = false, bool log = false, bool toUC=true);

void circular_shift(cv::InputArray src, cv::OutputArray dst, cv::Point delta) {
  cv::Mat _src = src.getMat();
  dst.create(_src.size(), _src.type());
  cv::Mat _dst = dst.getMat();

  int delta_x = delta.x % (_src.size().width - 1);
  int delta_y = delta.y % (_src.size().height - 1);

  int idx_x = delta_x >= 0 ? delta_x : _src.size().width + delta_x;
  int idx_y = delta_y >= 0 ? delta_y : _src.size().height + delta_y;

  cv::Mat tmp(_src.size(), _src.type());

  _src(cv::Rect(0, 0, _dst.size().width - idx_x, _dst.size().height))
      .copyTo(tmp(
          cv::Rect(idx_x, 0, _dst.size().width - idx_x, _dst.size().height)));

  _src(cv::Rect(_dst.size().width - idx_x, 0, idx_x, _dst.size().height))
      .copyTo(tmp(cv::Rect(0, 0, idx_x, _dst.size().height)));

  tmp(cv::Rect(0, 0, _dst.size().width, _dst.size().height - idx_y))
      .copyTo(_dst(
          cv::Rect(0, idx_y, _dst.size().width, _dst.size().height - idx_y)));

  tmp(cv::Rect(0, _dst.size().height - idx_y, _dst.size().width, idx_y))
      .copyTo(_dst(cv::Rect(0, 0, _dst.size().width, idx_y)));
}

void fft_shift(cv::InputArray src, cv::OutputArray dst) {
  cv::Mat _src = src.getMat();
  cv::Point delta;
  delta.x = _src.size().width / 2;
  delta.y = _src.size().height / 2;
  circular_shift(src, dst, delta);
}

void ifft_shift(cv::InputArray src, cv::OutputArray dst) {
  cv::Mat _src = src.getMat();
  cv::Point delta;
  delta.x = -_src.size().width / 2;
  delta.y = -_src.size().height / 2;
  circular_shift(src, dst, delta);
}

void FT(cv::InputArray src, cv::OutputArray dst, bool _mag, bool log) {
  cv::Mat _src = src.getMat();

  int M = cv::getOptimalDFTSize(_src.rows);
  int N = cv::getOptimalDFTSize(_src.cols);
  cv::Mat padded;
  cv::copyMakeBorder(_src, padded, 0, M - _src.rows, 0, N - _src.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::Mat planes[] = {cv::Mat_<float>(padded),
                      cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg);
  cv::dft(complexImg, complexImg);
  if (_mag) {
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    if (log) {
      mag += cv::Scalar::all(1);
      cv::log(mag, mag);
    }
    dst.create(cv::Size(mag.cols & -2, mag.rows & -2), CV_32FC1);
    cv::Mat _dst = dst.getMat();
    mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2)).copyTo(_dst);
  } else {
    dst.create(cv::Size(complexImg.cols & -2, complexImg.rows & -2), CV_32FC2);
    cv::Mat _dst = dst.getMat();
    complexImg(cv::Rect(0, 0, complexImg.cols & -2, complexImg.rows & -2))
        .copyTo(_dst);
  }
}

void freq(cv::InputArray src, cv::OutputArray dst, bool shift, bool mag,
          bool log, bool toUC) {
  cv::Mat _src = src.getMat();
  CV_Assert(_src.type() == CV_8UC1);
  cv::Mat tmp;
  FT(_src, tmp, mag, log);
  if (shift)
    fft_shift(tmp, tmp);
  if (mag) {
    if (toUC) {
      cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
      dst.create(tmp.size(), CV_8UC1);
      cv::Mat _dst = dst.getMat();
      tmp.convertTo(_dst, CV_8UC1);
    } else {
      dst.create(tmp.size(), CV_32FC1);
      cv::Mat _dst = dst.getMat();
      tmp.copyTo(_dst);
    }
  } else {
    dst.create(tmp.size(), CV_32FC2);
    cv::Mat _dst = dst.getMat();
    tmp.copyTo(_dst);
  }
}

} // namespace ex3

#endif /* EX3_HPP */