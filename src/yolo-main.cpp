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

#include <iostream>
#include <ostream>
#include <vector>

#include "yolo-dataset.hpp"
#include "yolo-loss.hpp"
#include "yolo-model.hpp"

int main()
{
  // TODO: add check for device and create accordingly.
  torch::Device device = torch::Device(torch::kCPU);
  std::string data_set_root{ "training/data/PASCAL_VOC" };

  // Create the module install
  YOLOv3 yolov3{ 3, 20, 3 };
  yolov3->to(device);

  torch::optim::Adam optimizer(yolov3->parameters(), torch::optim::AdamOptions(1e-3));

  // Preaper the data set
  YOLODataset y_data_set{ data_set_root, YOLODataset::Mode::kTrain, 13 };
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      y_data_set, torch::data::DataLoaderOptions().batch_size(16).workers(10));

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