#include <torch/torch.h>
#include <vector>
// The batch size for training.
const int64_t kBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

const int64_t kNumberOfTestEpochs = 5;

// Where to find the MNIST dataset.
const char *kDataFolder = "./training/data/mnist";

const char *kCheckPointFolder = "./training/check_points/simple_mnist";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 500;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

const bool kTest = true;

const int64_t kTestBatchSize = 1;

const bool kTrain = true;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

//
//
#include "helper.hpp"
int main() {
  torch::manual_seed(1);

  torch::DeviceType device_type;

  device_type = torch::kMPS;

  torch::Device device(device_type);

  torch::nn::Sequential module = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)), torch::nn::ReLU(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
      torch::nn::Dropout(),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
      torch::nn::Flatten(), torch::nn::ReLU(),
      torch::nn::Linear(torch::nn::LinearOptions(320, 50)), torch::nn::ReLU(),
      torch::nn::Dropout(torch::nn::DropoutOptions(0.5)),
      torch::nn::Linear(torch::nn::LinearOptions(50, 10)),
      torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
  module->to(device);
  torch::optim::SGD resnet_optimizer(
      module->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  std::cout << "Module is built " << std::endl;
  std::cout << "============================================================="
            << std::endl;
  std::cout << module << std::endl;
  std::cout << "============================================================="
            << std::endl;
  if (kRestoreFromCheckpoint) {
    torch::load(module,
                std::string(kCheckPointFolder) + "/resnet-checkpoint.pt");
    torch::load(resnet_optimizer, std::string(kCheckPointFolder) +
                                      "/resnet_optimizer-checkpoint.pt");
    std::cout << "Loding network weights completed" << std::endl;
  }

  if (kTrain) {
    auto train_data =
        torch::data::datasets::MNIST(kDataFolder,
                                     torch::data::datasets::MNIST::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    train(module, resnet_optimizer, device, train_data);
  }

  if (kTest) {
    auto test_data = torch::data::datasets::MNIST(
                         kDataFolder, torch::data::datasets::MNIST::Mode::kTest)
                         .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                         .map(torch::data::transforms::Stack<>());
    validate(module, device, test_data);
  }

  return 0;
}
