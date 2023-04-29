#include <sys/types.h>

#include <cmath>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "opencv2/core.hpp"

class Target
{
 public:
  enum class Type
  {
    MID_POINT,
    TOP_LEFT
  };
  Target()
  {
  }
  Target(float x, float y, float w, float h, size_t c, std::string c_name = "None", Type t = Type::MID_POINT)
      : x{ x }, y{ y }, w{ w }, h{ h }, c{ c }, c_name{ c_name }, type(t)
  {
  }
  float x{}, y{}, w{}, h{};
  size_t c{};
  std::string c_name{};
  Type type{ Type::MID_POINT };
};

inline cv::Mat& mark_target(cv::Mat& image, const Target& target, size_t s = 7)
{
  auto x_opncv = target.x - target.w / 2;
  auto y_opncv = target.y - target.h / 2;
  cv::Rect rect(x_opncv, y_opncv, target.w, target.h);
cv:
  cv::circle(image, cv::Point(target.x, target.y), 3, cv::Scalar(255, 0, 0), 8);
  cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2, 8, 0);
  cv::putText(
      image, target.c_name, cv::Point(x_opncv, y_opncv), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
  uint i, j;
  i = floor(target.x / (s * s));
  j = floor(target.y / (s * s));
  std::string cell_name{ "i= " + std::to_string(i) + " j= " + std::to_string(j) };
  cv::putText(image, cell_name, cv::Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
  return image;
}

inline cv::Mat& mark_cells(cv::Mat& image, const size_t s = 7)
{
  for (int i = 0; i < s; i++)
  {
    for (int j = 0; j < s; j++)
    {
      auto cell_width = image.size().width / s;
      auto cell_height = image.size().height / s;
      cv::Rect rect(cell_width * i, cell_height * j, cell_height, cell_height);
      cv::rectangle(image, rect, cv::Scalar(255, 255, 255), 1, 1, 0);
    }
  }
  return image;
}

inline void display_image(const cv::Mat& image)
{
  cv::imshow("test", image);
  cv::waitKey(1);
}

inline bool is_quit()
{
  return (cv::waitKey(0) == 'q');
}