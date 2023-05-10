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

#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string_view>
#include <vector>

#include "toml++/toml.h"
#include "utils/open-cv-helper.hpp"
#include "utils/matplot.hpp"
#include "yolo-dataset.hpp"
#include "yolo-loss.hpp"
#include "yolo-model.hpp"

bool app_run = true;
void signalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received" << std::endl;
  app_run = false;
}

int main()
{
  // register signal SIGINT and signal handler
  signal(SIGINT, signalHandler);

  // create a data ploter for loss
  DataPloter data_loter;

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

  auto model_config = config["yolo_model"];
  std::filesystem::path model_save_directory{ model_config["model_path"].value<std::string>().value() };
  std::filesystem::create_directory(model_save_directory);
  std::filesystem::path model_weight_file_name{ model_config["model_weight_file_name"].value<std::string>().value() };
  std::filesystem::path model_save_file_path = model_save_directory / model_weight_file_name;
  std::filesystem::path model_loss_graph_file_name{ model_config["model_loss_graph_file"].value<std::string>().value() };
  std::filesystem::path model_loss_graph_file_path = model_save_directory / model_loss_graph_file_name;

  // Create the module from fonfiguration
  YOLOv3 yolov3{ model_config };
  if (std::filesystem::exists(model_save_file_path))
  {
    torch::load(yolov3, model_save_file_path);
  }
  yolov3->to(device);

  std::cout << "Yolo model created" << std::endl;

  auto training_config = config["training_loop"];
  double learning_rate = training_config["learning_rate"].value<double>().value();
  double momentum = training_config["momentum"].value<double>().value();
  torch::optim::Adam optimizer(yolov3->parameters(), torch::optim::AdamOptions().lr(learning_rate));

  // Preaper the data set
  uint64_t batch_size = training_config["batch_size"].value<uint64_t>().value();
  uint64_t number_of_workers = training_config["number_of_workers"].value<uint64_t>().value();

  YOLODataset y_data_set{ YOLODataset::Mode::kTrain, config["data_set"] };
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      y_data_set, torch::data::DataLoaderOptions().batch_size(batch_size).workers(number_of_workers));

  std::cout << "Data loader created" << std::endl;

  // Create the loss object
  YOLOLoss yolo_loss{};

  std::cout << "Start training" << std::endl;

  float avarage_loss = 0;
  size_t epochs = training_config["epochs"].value<size_t>().value();
  bool run = true;
  bool display = training_config["display"].value<bool>().value();
  for (size_t i = 0; i < epochs && run; i++)
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

      // Do the prediction
      auto model_prediction_tensor = yolov3(batch_inputs_tensor);
      // Compute the loss
      auto loss = yolo_loss(model_prediction_tensor, batch_targets_tensor);
      // Reset gradients.
      optimizer.zero_grad();
      // Compute gradients for all trainable weights
      loss.backward();
      // Update the weights with gradients
      optimizer.step();

      auto loss_data = loss.sum().data().item<float>();
      avarage_loss += loss_data;
      avarage_loss = avarage_loss / 2;
      ++batch_count;
      data_loter.add_data(loss_data);
      std::cout << "Epoch " << i << " Avarage_loss = " << avarage_loss << " Batch count = " << batch_count
                << " loss = " << loss_data << std::endl;
      if (display)
      {
        run = display_imgae(batch_inputs_tensor[0], model_prediction_tensor[0], batch_targets_tensor[0]);
        run = run && !data_loter.show_plot();
      }

      run = app_run;

      if (!run)
      {
        break;
      }
    }
  }

  std::cout << "Done iterating" << std::endl;
  torch::save(yolov3, model_save_file_path);
  std::cout << "Saved the model! " << model_save_file_path << std::endl;
  data_loter.save_graph(model_loss_graph_file_path);

  return 0;
}