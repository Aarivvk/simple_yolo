#include <torch/torch.h>

#include <iostream>
#include <opencv2/highgui.hpp>
#include <ostream>

#include "opencv2/core.hpp"

struct Settings
{
  int64_t kBatchSize;
  int64_t kNumberOfEpochs;
  int64_t kLogInterval;
  int64_t kTestBatchSize;
  int64_t kNumberOfTestEpochs;
};

template<class Module, class Optimizer, class Device, class DataSet>
void train(Module &module, Optimizer &optimizer, Device &device, DataSet data_set, const Settings settings)
{
  std::cout << "started the training..." << std::endl;
  // Assume the MNIST dataset is available under `mnist`;
  auto dataset = data_set;
  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(settings.kBatchSize));

  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), settings.kBatchSize);
  std::cout << "Lodded training data set." << std::endl;
  int64_t checkpoint_counter = 0;

  for (int64_t epoch = 1; epoch <= settings.kNumberOfEpochs; ++epoch)
  {
    module->train();
    int64_t batch_index = 0;
    for (torch::data::Example<> &batch : *data_loader)
    {
      optimizer.zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = batch.target.to(device);
      torch::Tensor real_output = module->forward(real_images);
      torch::Tensor loss = torch::cross_entropy_loss(real_output, real_labels);
      loss.backward();   // calculate the gradiants [the resulting tensor holds the compute graph].
      optimizer.step();  // Apply the calculated grads.
      batch_index++;
      if (batch_index % settings.kLogInterval == 0)
      {
        std::printf(
            "\r[%2ld/%2ld][%3ld/%3ld] | loss: %.4f",
            epoch,
            settings.kNumberOfEpochs,
            batch_index,
            batches_per_epoch,
            loss.item<float>());
      }
    }
  }
  std::cout << " Training complete!" << std::endl;
}

template<class Module, class Device, class DataSet>
void validate(Module &module, Device &device, DataSet data_set, const Settings settings)
{
  std::cout << "Started Testing" << std::endl;
  auto test_dataset = data_set;
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), settings.kTestBatchSize);
  std::cout << "Loading Test data is complete" << std::endl;
  for (int64_t epoch = 1; epoch <= settings.kNumberOfTestEpochs; ++epoch)
  {
    module->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto &batch : *test_loader)
    {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);
      auto output = module->forward(data);
      test_loss += torch::cross_entropy_loss(output, targets, {}, torch::Reduction::Sum).template item<float>();
      correct += output.argmax(1).eq(targets).sum().template item<int64_t>();
    }

    test_loss /= test_dataset_size;
    std::printf(
        "\nTest set: Average loss: %.4f | Accuracy: %.3f\n", test_loss, static_cast<double>(correct) / test_dataset_size);
  }
}

template<class Module, class Device, class DataSet>
void demo(Module &module, Device &device, DataSet data_set)
{
  std::cout << "Started Testing" << std::endl;
  auto test_dataset = data_set;
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1);
  std::cout << "Loading Test data is complete" << std::endl;
  module->eval();
  for (const auto &batch : *test_loader)
  {
    auto data = batch.data.to(device);
    torch::Tensor image = data.squeeze().squeeze().detach().permute({ 0, 1 });
    image = image.mul(255).clamp(0, 255).to(torch::kU8);
    image = image.to(torch::kCPU);
    cv::Mat resultImg(image.sizes()[0], image.sizes()[1], 1);
    std::memcpy((void *)resultImg.data, image.data_ptr(), sizeof(torch::kU8) * image.numel());
    cv::imshow("test", resultImg);

    auto targets = batch.target.to(device);
    auto output = module->forward(data);
    auto ret = output.argmax(1);
    std::cout << "Prediction " << ret << std::endl;

    if (cv::waitKey(0) == 'q')
    {
      break;
    }
  }
}

// torch::Tensor out_tensor = module->forward(inputs).toTensor();
// assert(out_tensor.device().type() == torch::kCUDA);
// out_tensor=out_tensor.squeeze().detach().permute({1,2,0});
// out_tensor=out_tensor.mul(255).clamp(0,255).to(torch::kU8);
// out_tensor=out_tensor.to(torch::kCPU);
// cv::Mat resultImg(512, 512,CV_8UC3);
// std::memcpy((void*)resultImg.data,out_tensor.data_ptr(),sizeof(torch::kU8)*out_tensor.numel());