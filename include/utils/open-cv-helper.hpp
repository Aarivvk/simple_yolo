#ifndef FEC996AE_B209_4263_A08F_FA1D042D8E91
#define FEC996AE_B209_4263_A08F_FA1D042D8E91
#include <sys/types.h>

#include <cmath>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "opencv2/core.hpp"

// https :  // docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
cv::VideoCapture get_camera(int width, int height)
{
  cv::Mat frame;
  cv::VideoCapture cap;
  int deviceID = 0;         // 0 = open default camera
  int apiID = cv::CAP_ANY;  // 0 = autodetect default API
  // open selected camera using selected API
  cap.open(deviceID, apiID);
  // check if we succeeded
  if (!cap.isOpened())
  {
    std::cerr << "ERROR! Unable to open camera\n";
    exit(1);
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  return cap;
}

std::vector<int> get_selected_indexes(torch::Tensor predictions)
{
  auto classes = predictions.slice(2, torch::indexing::None, 20, 1);
  auto classess_flaten = classes.flatten(0, 1);

  auto objectness = predictions.slice(2, 24, torch::indexing::None, 1);
  auto objectness_flaten = objectness.flatten(0, 1).squeeze();


  std::vector<int> selected_index{};
  for (size_t i = 0; i < classess_flaten.size(0); i++)
  {
    auto class_indexe = classess_flaten[i].argmax().item<int>();
    auto calss_prob = classess_flaten[i][class_indexe].item<float>();
    auto objectness_prob = objectness_flaten[i].item<float>();
    if(objectness_prob > 0.5 && calss_prob > 0.5 ){
    std::cout << "Selecting the index " << i << " with objectness_prob " << objectness_prob << " calss_prob " << calss_prob
              << " class_index " << class_indexe << std::endl;
    selected_index.push_back(i);
    }
  }

  return selected_index;
}

void draw_bounding_box(torch::Tensor& prediction, cv::Mat& frame, bool r = true)
{
  std::cout << "____________________________________________________" << std::endl;
  std::vector<int> selected_index = get_selected_indexes(prediction);
  auto bounding_box = prediction.slice(2, 20, 25, 1).flatten(0, 1);
  bounding_box = bounding_box.sigmoid();

  // Scale up the bounding box
  auto x = bounding_box.slice(1, 0, 1, 1);
  auto y = bounding_box.slice(1, 1, 2, 1);
  auto w = bounding_box.slice(1, 2, 3, 1);
  auto h = bounding_box.slice(1, 3, 4, 1);

  x = x * frame.cols;
  y = y * frame.rows;
  w = w * frame.cols;
  h = h * frame.rows;

  auto rect_x1 = x - w/2;
  auto rect_y1 = y - h/2;
  auto rect_x2 = x + w/2;
  auto rect_y2 = y + h/2;

  for (auto& item : selected_index)
  {
    int x1 = static_cast<int>(rect_x1[item].item<int>());
    int y1 = static_cast<int>(rect_y1[item].item<int>());
    int x2 = static_cast<int>(rect_x2[item].item<int>());
    int y2 = static_cast<int>(rect_y2[item].item<int>());
    if (r)
    {
      cv::rectangle(frame, { x1, y1 }, { x2, y2 }, { 255, 0, 0 }, 3);
    }
    else
    {
      cv::rectangle(frame, { x1, y1 }, { x2, y2 }, { 0, 0, 255 }, 4);
    }
  }
  std::cout << "____________________________________________________" << std::endl;
  std::cout << std::endl;
}

cv::Mat get_cv_frame(torch::Tensor t_image)
{
  t_image = t_image.squeeze().detach().permute({ 1, 2, 0 }).contiguous();
  t_image = t_image.mul(255).clamp(0, 255).to(torch::kByte).to(torch::kCPU);
  int width = t_image.size(0);
  int height = t_image.size(1);
  int channels = t_image.size(2);

  cv::Mat output_mat(cv::Size{ width, height }, CV_8UC3, t_image.data_ptr<uchar>());
  return output_mat.clone();
}

bool display_imgae(cv::Mat& frame)
{
  // Display the image
  cv::imshow("Simple YOLO", frame);
  if (cv::waitKey(5) >= 0)
  {
    return false;
  }
  return true;
}

bool display_imgae(torch::Tensor t_image)
{
  cv::Mat image = get_cv_frame(t_image);
  draw_bounding_box(t_image, image);
  return display_imgae(image);
}

bool display_imgae(torch::Tensor t_image, torch::Tensor t_predict, torch::Tensor t_target)
{
  cv::Mat image = get_cv_frame(t_image);
  draw_bounding_box(t_target, image);
  draw_bounding_box(t_predict, image, false);
  return display_imgae(image);
}

#endif /* FEC996AE_B209_4263_A08F_FA1D042D8E91 */
