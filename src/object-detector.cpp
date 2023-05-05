#include <ATen/core/TensorBody.h>
#include <ATen/ops/cat.h>

#include <cstddef>
#include <filesystem>
#include <opencv2/core/mat.hpp>

#include "torch/torch.h"
#include "utils/open-cv-helper.hpp"
#include "yolo-model.hpp"

int main(void)
{
  // Select the device
  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "Using CUDA device" << std::endl;
  }
  else
  {
    std::cout << "Using CPU device" << std::endl;
  }

  // Load the configuration
  std::filesystem::path config_directory("config");
  std::filesystem::path config_file("darknet.toml");
  std::filesystem::path config_file_path = config_directory / config_file;
  std::cout << "Loading the configuration file " << config_file_path.string() << std::endl;
  auto config = toml::parse_file(config_file_path.string());

  // Create the module from configurations
  YOLOv3 yolov3{ config["yolo_model"] };
  std::filesystem::path model_save_directory{ "saved_models" };
  std::filesystem::create_directory(model_save_directory);
  std::filesystem::path model_weight_file_name{ "yolv3.pt" };
  std::filesystem::path model_weight_file_path = model_save_directory / model_weight_file_name;
  torch::load(yolov3, model_weight_file_path);

  // Disable gradient calculation
  torch::NoGradGuard no_grad;
  // set for prediction
  yolov3->eval();
  // Move model to the device.
  yolov3->to(device);

  // Preaper the camera feed.
  int width = config["camera"]["image_width"].value<int>().value();
  int height = config["camera"]["image_height"].value<int>().value();
  auto cap = get_camera(width, height);

  // Extract the bounding boxes and class probabilities

  // Draw the bounding box and class on the original image with confidance.

  int width_r = config["training"]["image_width"].value<int>().value();
  int height_r = config["training"]["image_height"].value<int>().value();
  cv::Mat frame, r_frame;
  bool run{true};
  while (run)
  {
    cap.read(frame);
    if (frame.empty())
    {
      std::cerr << "ERROR! blank frame grabbed\n";
    }

    cv::resize(frame, frame, { width_r, height_r });

    // Convert the cv::Mat to torch::tensor
    auto img_tensor = torch::from_blob(frame.data, { frame.rows, frame.cols, frame.channels() }, torch::kByte).clone();
    img_tensor = img_tensor.permute({ 2, 0, 1 });
    img_tensor = img_tensor.to(torch::kF32);
    img_tensor = img_tensor.unsqueeze(0);
    img_tensor = img_tensor.to(device);
    img_tensor = img_tensor / 255;

    // Feed to the simple-yolo model
    auto output = yolov3(img_tensor);
    output = output.squeeze();
    // extract the class index
    draw_bounding_box(output, frame, false);

    run = display_imgae(frame);
  }

  std::cout << "Program terminated" << std::endl;
}