#ifndef FEC996AE_B209_4263_A08F_FA1D042D8E91
#define FEC996AE_B209_4263_A08F_FA1D042D8E91
#include <sys/types.h>
#include <torch/nn/modules/activation.h>

#include <cmath>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "toml++/toml.h"

#include "opencv2/core.hpp"

// https :  // docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
cv::VideoCapture get_camera(int width, int height)
{
  cv::Mat frame;
  cv::VideoCapture cap(0);
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

std::vector<int> get_selected_indexes(torch::Tensor predictions, bool target_draw, toml::node_view<toml::node> config)
{
  auto classes = predictions.slice(2, torch::indexing::None, 20, 1);
  torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(1));
  auto classess_flaten = classes.flatten(0, 1);

  auto objectness = predictions.slice(2, 24, torch::indexing::None, 1);
  auto objectness_flaten = objectness.flatten(0, 1);
  if (!target_draw)
  {
    classess_flaten = softmax(classess_flaten);
    // objectness_flaten = objectness_flaten.sigmoid();
  }

  std::vector<int> selected_index{};
  for (size_t i = 0; i < objectness_flaten.size(0); i++)
  {
    auto class_indexe = classess_flaten[i].argmax().item<int>();
    auto calss_prob = classess_flaten[i][class_indexe].item<float>();
    auto objectness_prob = objectness_flaten[i].item<float>();
    auto obj_threshold = config["bojectness_threshold"].value<double>().value();
    auto class_threshold = config["class_threshold"].value<double>().value();
    if (objectness_prob >= obj_threshold && calss_prob > class_threshold)
    {
      // std::cout << "Selecting the index " << i << " with objectness_prob " << objectness_prob << " calss_prob " << calss_prob
      //           << " class_index " << class_indexe << std::endl;
      selected_index.push_back(i);
    }
  }

  return selected_index;
}

void draw_bounding_box(torch::Tensor& prediction, cv::Mat& frame, bool target_draw, toml::node_view<toml::node> config)
{
  std::string name;
  if (target_draw)
  {
    name = "Target";
  }
  else
  {
    name = "prediction";
  }
  // std::cout << std::endl << "_______________________" << name << "_____________________________" << std::endl;
  std::vector<int> selected_index = get_selected_indexes(prediction, target_draw, config);
  auto bounding_box = prediction.slice(2, 20, 25, 1).flatten(0, 1);

  auto classes = prediction.slice(2, torch::indexing::None, 20, 1);
  auto classess_flaten = classes.flatten(0, 1);
  
  if (!target_draw)
  {
    // bounding_box = bounding_box.sigmoid();
    torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(1));
    classess_flaten = softmax(classess_flaten);
  }

  // Scale up the bounding box
  auto x = bounding_box.slice(1, 0, 1, 1);
  auto y = bounding_box.slice(1, 1, 2, 1);
  auto w = bounding_box.slice(1, 2, 3, 1);
  auto h = bounding_box.slice(1, 3, 4, 1);

  x = x * frame.size().width;
  y = y * frame.size().height;
  w = w * frame.size().width;
  h = h * frame.size().height;

  auto rect_x1 = x - w / 2;
  auto rect_y1 = y - h / 2;
  auto rect_x2 = x + w / 2;
  auto rect_y2 = y + h / 2;

  for (auto& item : selected_index)
  {
    auto class_indexe = classess_flaten[item].argmax().item<int>();
    auto calss_prob = classess_flaten[item][class_indexe].item<float>();

    int x1 = static_cast<int>(rect_x1[item].item<int>());
    int y1 = static_cast<int>(rect_y1[item].item<int>());
    int x2 = static_cast<int>(rect_x2[item].item<int>());
    int y2 = static_cast<int>(rect_y2[item].item<int>());

    cv::Scalar color{ 0, 0, 255 };
    int thickness = 2;
    if (target_draw)
    {
      color = { 0, 255, 0 };
      thickness = 2;
    }
    cv::rectangle(frame, { x1, y1 }, { x2, y2 }, color, thickness);
    cv::circle(frame, { x[item].item<int>(), y[item].item<int>() }, 5, color, thickness);
    std::string class_number_probability = "id=" + std::to_string(class_indexe) + " c=" + std::to_string(calss_prob);
    cv::putText(frame, class_number_probability, { x1, y1 - 3 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, thickness);
  }
  // std::cout << "_________________________" << name << "___________________________" << std::endl;
  // std::cout << std::endl;
}

cv::Mat get_cv_frame(torch::Tensor t_image)
{
  t_image = t_image.squeeze().detach().permute({ 1, 2, 0 }).contiguous();
  t_image = t_image.mul(255).clamp(0, 255).to(torch::kByte).to(torch::kCPU);
  int width = t_image.size(0);
  int height = t_image.size(1);
  int channels = t_image.size(2);
  auto img_size = cv::Size{};
  img_size.width = width;
  img_size.height = height;
  cv::Mat output_mat(img_size, CV_8UC3, t_image.data_ptr<uchar>());
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

bool display_imgae(torch::Tensor t_image, toml::node_view<toml::node> config)
{
  cv::Mat image = get_cv_frame(t_image);
  draw_bounding_box(t_image, image, false, config);
  return display_imgae(image);
}

bool display_imgae(torch::Tensor t_image, torch::Tensor t_predict, torch::Tensor t_target, toml::node_view<toml::node> config)
{
  cv::Mat image = get_cv_frame(t_image);
  draw_bounding_box(t_target, image, true, config);
  draw_bounding_box(t_predict, image, false, config);
  return display_imgae(image);
}

#endif /* FEC996AE_B209_4263_A08F_FA1D042D8E91 */
