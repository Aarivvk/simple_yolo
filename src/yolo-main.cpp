#include <ATen/core/TensorBody.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/zero.h>
#include <ATen/ops/zeros.h>
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
#include <fstream>
#include <iostream>
#include <ostream>
#include <string_view>
#include <vector>

#include "toml++/toml.h"
#include "utils/matplot.hpp"
#include "utils/open-cv-helper.hpp"
#include "yolo-dataset.hpp"
#include "yolo-loss.hpp"
#include "yolo-model.hpp"

// https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/basics/pytorch_basics/main.cpp

bool app_run = true;
void signalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received" << std::endl;
  app_run = false;
}

int main()
{
  torch::manual_seed(3);
  // register signal SIGINT and signal handler
  signal(SIGINT, signalHandler);

  // create a data ploter for loss
  DataPloter data_ploter;

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
  std::filesystem::path model_weight_file_name_current{
    model_config["model_weight_file_name_current"].value<std::string>().value()
  };
  std::filesystem::path model_save_file_path = model_save_directory / model_weight_file_name;
  std::filesystem::path model_save_file_path_current = model_save_directory / model_weight_file_name_current;
  std::filesystem::path model_loss_graph_file_name{ model_config["model_loss_graph_file"].value<std::string>().value() };
  std::filesystem::path model_loss_graph_file_name_current{
    model_config["model_loss_graph_file_current"].value<std::string>().value()
  };
  std::filesystem::path model_optimize_file_name{ model_config["model_optimize_file_name"].value<std::string>().value() };
  std::filesystem::path model_loss_graph_file_path = model_save_directory / model_loss_graph_file_name;
  std::filesystem::path model_loss_graph_file_path_current = model_save_directory / model_loss_graph_file_name_current;

  // Create the module from fonfiguration
  YOLOv3 yolov3{ model_config };
  // Create the loss object
  YOLOLoss yolo_loss{};

  auto training_config = config["training_loop"];
  double learning_rate = training_config["learning_rate"].value<double>().value();
  double momentum = training_config["momentum"].value<double>().value();
  double weight_decay = training_config["weight_decay"].value<double>().value();
  torch::optim::AdamW optimizer(
      yolov3->parameters(), torch::optim::AdamWOptions().lr(learning_rate).weight_decay(weight_decay));
  if (std::filesystem::exists(model_save_file_path))
  {
    torch::load(yolov3, model_save_file_path);
    torch::load(optimizer, model_save_directory / model_optimize_file_name);
    std::cout << "Loaded previous weights from " << model_save_file_path << std::endl;
  }
  yolov3->to(device);

  std::cout << "Yolo model created" << std::endl;

  // Preaper the data set
  uint64_t batch_size = training_config["batch_size"].value<uint64_t>().value();
  uint64_t number_of_workers = training_config["number_of_workers"].value<uint64_t>().value();

  YOLODataset train_data_set{ YOLODataset::Mode::kTrain, config["data_set"] };
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_data_set, torch::data::DataLoaderOptions().batch_size(batch_size).workers(number_of_workers));
  size_t train_data_size = train_data_set.size().value();

  YOLODataset test_data_set{ YOLODataset::Mode::kTest, config["data_set"] };
  auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      test_data_set, torch::data::DataLoaderOptions().batch_size(batch_size).workers(number_of_workers));
  size_t test_data_size = test_data_set.size().value();
  std::cout << "Data loader created" << std::endl;

  std::cout << "Start training" << std::endl;
  std::cout << std::endl << std::endl << std::endl << std::endl;

  size_t epochs = training_config["epochs"].value<size_t>().value();
  bool run = true;
  bool display = training_config["display"].value<bool>().value();
  std::ofstream ofs(model_save_directory / config_file);
  ofs << config << std::flush;
  ofs.close();

  std::cout << std::fixed;
  yolov3->train();
  yolo_loss->train();
  double previous_loss{ 100 };

  size_t diverge_count{};
  for (size_t i = 0; i < epochs && run; i++)
  {
    // Iterate the data loader to yield batches from the dataset.
    torch::Tensor epoch_loss = torch::zeros({ 1 }).to(device);
    torch::Tensor epoch_precision = torch::zeros({ 1 }).to(device);
    size_t train_batch_count = 1;
    yolov3->train();
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
      auto precision = yolo_loss->accuracy(model_prediction_tensor, batch_targets_tensor);

      auto loss_data = loss.sum().item<double>();
      std::cout << '\r' << "\033[2K"
                << "[Train]"
                << "[" << i + 1 << "/" << epochs << "] [" << train_batch_count * batch_size << "/" << train_data_size << "] "
                << "Loss = " << loss_data << " | "
                << "Precision = " << precision << std::flush;
      torch::Tensor local_loss = torch::zeros({ 1 }).to(device);
      local_loss = local_loss + loss_data;
      epoch_loss = torch::cat({ epoch_loss, local_loss });
      torch::Tensor local_precision = torch::zeros({ 1 }).to(device);
      local_precision = local_precision + precision;
      epoch_precision = torch::cat({ epoch_precision, local_precision });

      if (display)
      {
        torch::NoGradGuard no_grad;
        run =
            display_imgae(batch_inputs_tensor[0], model_prediction_tensor[0], batch_targets_tensor[0], config["inference"]);
      }
      run = run && app_run;
      if (!run)
      {
        break;
      }
      ++train_batch_count;
    }
    data_ploter.add_data_train(epoch_loss.mean().data().item<double>(), i, epoch_precision.mean().data().item<double>());

    // Save weights for every epoch
    torch::save(yolov3, model_save_file_path_current);
    torch::save(optimizer, model_save_directory / model_optimize_file_name);
    data_ploter.save_graph(model_loss_graph_file_path_current);

    std::cout << std::endl;

    {
      torch::NoGradGuard no_grad;
      // Create the module from last saved weight
      YOLOv3 yolov3_test{ model_config };
      if (std::filesystem::exists(model_save_file_path_current))
      {
        torch::load(yolov3_test, model_save_file_path_current);
      }
      else
      {
        std::cout << "Error: no file to load " << model_save_file_path_current << std::endl;
        exit(1);
      }

      yolov3_test->to(device);
      yolov3_test->eval();
      // Create test loss
      YOLOLoss yolo_loss_test{};
      yolo_loss_test->to(device);
      yolo_loss_test->eval();

      torch::Tensor epoch_loss_validation = torch::zeros({ 1 }).to(device);
      torch::Tensor epoch_precision_validation = torch::zeros({ 1 }).to(device);
      size_t validation_batch_count = 1;
      for (auto &batch : *test_data_loader)
      {
        // https://discuss.pytorch.org/t/converting-std-vector-example-into-an-input-tensor-in-c/81549
        std::vector<torch::Tensor> batch_inputs_vector, batch_targets_vector;
        std::transform(batch.begin(), batch.end(), std::back_inserter(batch_inputs_vector), [](auto &e) { return e.data; });
        std::transform(
            batch.begin(), batch.end(), std::back_inserter(batch_targets_vector), [](auto &e) { return e.target; });
        torch::Tensor batch_inputs_tensor = torch::stack(batch_inputs_vector).to(device);
        torch::Tensor batch_targets_tensor = torch::stack(batch_targets_vector).to(device);

        // Do the prediction
        auto model_prediction_tensor = yolov3_test(batch_inputs_tensor);
        // Compute the loss
        auto loss = yolo_loss_test(model_prediction_tensor, batch_targets_tensor);
        auto precision = yolo_loss_test->accuracy(model_prediction_tensor, batch_targets_tensor);

        auto loss_data = loss.item<float>();

        std::cout << '\r' << "\033[2K"
                  << "[Test]"
                  << "[" << i + 1 << "/" << epochs << "] [" << validation_batch_count * batch_size << "/" << test_data_size
                  << "] "
                  << "Loss = " << loss_data << " | "
                  << "Precision = " << precision << std::flush;

        torch::Tensor local_loss = torch::zeros({ 1 }).to(device);
        local_loss = local_loss + loss_data;
        epoch_loss_validation = torch::cat({ epoch_loss_validation, local_loss });
        torch::Tensor local_precision = torch::zeros({ 1 }).to(device);
        local_precision = local_precision + precision;
        epoch_precision_validation = torch::cat({ epoch_precision_validation, local_precision });

        if (display)
        {
          run = display_imgae(
              batch_inputs_tensor[0], model_prediction_tensor[0], batch_targets_tensor[0], config["inference"]);
        }
        run = run && app_run;
        if (!run)
        {
          break;
        }
        ++validation_batch_count;
      }
      auto current_loss = epoch_loss_validation.mean().data().item<double>();
      data_ploter.add_data_validation(current_loss, i, epoch_precision_validation.mean().data().item<double>());

      // Save the configuration on dereasing test loss
      if (current_loss <= previous_loss && run)
      {
        std::cout << std::endl
                  << std::endl
                  << "Saving the best weight for loss " << current_loss << " epoch " << i + 1 << std::endl;
        torch::save(yolov3, model_save_file_path);
        data_ploter.save_graph(model_loss_graph_file_path);
        previous_loss = current_loss;
        diverge_count = 0;
      }
      else{
        diverge_count++;
        if(diverge_count > 3)
        {
          // Early stoping.
          std::cout << "Early stoping " << i << std::endl;
          run = false;
        }
      }
    }

    std::cout << std::endl << std::endl;

    if (display)
    {
      run = run && !data_ploter.show_plot();
    }
  }

  std::cout << "Done iterating" << std::endl;

  return 0;
}