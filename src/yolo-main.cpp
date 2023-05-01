#include <ATen/core/TensorBody.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/zero.h>
#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/data/dataloader_options.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/options/conv.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string_view>
#include <vector>

#include "utils/open-cv-helper.hpp"

#include "toml++/toml.h"
#include "yolo-dataset.hpp"
#include "yolo-loss.hpp"
#include "yolo-model.hpp"

int main()
{
  // TODO: add check for device and create accordingly.
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

  std::filesystem::path config_directory("config");
  std::filesystem::path config_file("darknet.toml");
  std::filesystem::path config_file_path = config_directory / config_file;
  std::cout << "Loading the configuration file " << config_file_path.string() << std::endl;
  auto config = toml::parse_file(config_file_path.string());

  // Create the module install
  YOLOv3 yolov3{ config["yolo_model"] };
  yolov3->to(device);

  auto training_config = config["training"];
  double learning_rate = training_config["learning_rate"].value<double>().value();
  torch::optim::Adam optimizer(yolov3->parameters(), torch::optim::AdamOptions(learning_rate));

  // Preaper the data set
  uint64_t batch_size = training_config["batch_size"].value<uint64_t>().value();
  uint64_t number_of_workers = training_config["number_of_workers"].value<uint64_t>().value();
  int number_of_detections = training_config["number_of_detections"].value<int>().value();
  int image_width = training_config["image_width"].value<int>().value();
  int image_height = training_config["image_height"].value<int>().value();
  const std::string train_data_set_root = training_config["training_data_directory"].value<std::string>().value();
  std::cout << "train_data_set_root " << train_data_set_root << std::endl;

  YOLODataset y_data_set{ train_data_set_root, YOLODataset::Mode::kTrain, number_of_detections, image_width, image_height };
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      y_data_set, torch::data::DataLoaderOptions().batch_size(batch_size).workers(number_of_workers));

  // Create the loss object
  YOLOLoss yolo_loss{};

  std::cout << "Start training" << std::endl;

  // Set the model in training mode
  yolov3->train();

  float avarage_loss = 0;
  size_t epochs = training_config["epochs"].value<size_t>().value();

  for (size_t i = 0; i < epochs; i++)
  {
     yolov3->train();
    int64_t batch_count = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto &batch : *train_data_loader)
    {
      // https://discuss.pytorch.org/t/converting-std-vector-example-into-an-input-tensor-in-c/81549
      std::vector<torch::Tensor> batch_inputs_vector, batch_targets_vector;
      std::transform(batch.begin(), batch.end(), std::back_inserter(batch_inputs_vector), [](auto &e) { return e.data; });
      std::transform(batch.begin(), batch.end(), std::back_inserter(batch_targets_vector), [](auto &e) { return e.target; });
      torch::Tensor batch_inputs_tensor = torch::stack(batch_inputs_vector).to(device);
      torch::Tensor batch_targets_tensor = torch::stack(batch_targets_vector).to(device);

      // Reset gradients.
      optimizer.zero_grad();
      // Do the prediction
      auto model_prediction_tensor = yolov3(batch_inputs_tensor);
      // Compute the loss
      auto loss = yolo_loss(model_prediction_tensor, batch_targets_tensor);

      // Compute gradients for all trainable weights
      loss.backward();

      // Update the weights with gradients
      optimizer.step();

      auto loss_data = loss.data().item<float>();
      avarage_loss += loss_data;
      avarage_loss = avarage_loss / 2;
      ++batch_count;

      std::cout << "Epoch " << i << " Avarage_loss = " << avarage_loss << " Batch count = " << batch_count
                << " loss = " << loss_data << std::endl;

      // bool ret = display_imgae(batch_inputs_tensor[0], model_prediction_tensor[0], batch_targets_tensor[0]);
    }
  }

  std::filesystem::path model_save_directory{ "saved_models" };
  std::filesystem::create_directory(model_save_directory);
  std::filesystem::path model_weight_file_name{ "yolv3.pt" };
  std::filesystem::path model_save_file_path = model_save_directory / model_weight_file_name;

  // Save the trained module for later reuse.
  std::cout << "Saved the model! " << model_save_file_path << std::endl;
  torch::save(yolov3, model_save_file_path);

  std::cout << "Done iterating" << std::endl;

  return 0;
}