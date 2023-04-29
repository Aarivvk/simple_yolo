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

#include "yolo-dataset.hpp"
#include "yolo-loss.hpp"
#include "yolo-model.hpp"

#include "toml++/toml.h"

int main()
{
  // TODO: add check for device and create accordingly.
  torch::Device device = torch::Device(torch::kCPU);
  std::string data_set_root{ "training/data/PASCAL_VOC" };

  std::filesystem::path config_directory("config");
  std::filesystem::path config_file("darknet.toml");
  std::filesystem::path config_file_path = config_directory / config_file;
  auto config = toml::parse_file(std::string_view(config_file_path.string()));

  // Create the module install
  YOLOv3 yolov3{config["yolo_model"]};
  yolov3->to(device);

  auto training_config = config["training"];
  double learning_rate = training_config["learning_rate"].value<double>().value();
  torch::optim::Adam optimizer(yolov3->parameters(), torch::optim::AdamOptions(learning_rate));

  // Preaper the data set
  uint64_t batch_size = training_config["batch_size"].value<uint64_t>().value();
  uint64_t number_of_workers = training_config["number_of_workers"].value<uint64_t>().value();
  int number_of_detections = training_config["number_of_detections"].value<int>().value();
  YOLODataset y_data_set{ data_set_root, YOLODataset::Mode::kTrain, number_of_detections };
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      y_data_set, torch::data::DataLoaderOptions().batch_size(batch_size).workers(number_of_workers));

  // Create the loss object
  YOLOLoss yolo_loss{};

  std::cout << "Start training" << std::endl;

  // Set the model in training mode
  yolov3->train();

  // Iterate the data loader to yield batches from the dataset.
  for (auto &batch : *train_data_loader)
  {
    // Reset gradients.
    optimizer.zero_grad();

    // https://discuss.pytorch.org/t/converting-std-vector-example-into-an-input-tensor-in-c/81549
    std::vector<torch::Tensor> batch_inputs_vector, batch_targets_vector;
    std::transform(batch.begin(), batch.end(), std::back_inserter(batch_inputs_vector), [](auto &e) { return e.data; });
    std::transform(batch.begin(), batch.end(), std::back_inserter(batch_targets_vector), [](auto &e) { return e.target; });
    torch::Tensor batch_inputs_tensor = torch::stack(batch_inputs_vector);
    torch::Tensor batch_targets_tensor = torch::stack(batch_targets_vector);

    auto model_prediction_tensor = yolov3(batch_inputs_tensor);

    auto loss = yolo_loss.compute_loss(model_prediction_tensor, batch_targets_tensor);

    // Compute gradients for all trainable weights
    loss.backward();

    // Update the weights with gradients
    optimizer.step();

    std::cout << "loss " << loss.data().item<float>() << std::endl;
  }

  std::filesystem::path model_save_directory{ "saved_models" };
  std::filesystem::path model_weight_file_name{ "yolv3.pt" };
  std::filesystem::path model_save_file_path = model_save_directory / model_weight_file_name;

  // Save the trained module for later reuse.
  torch::save(yolov3, model_save_file_path);

  std::cout << "Done iterating" << std::endl;

  return 0;
}