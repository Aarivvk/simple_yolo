#include "resnet.hpp"
#include "cifar10.hpp"
// #include "resnet_other.hpp"
#include <vector>
#include <chrono>

// The batch size for training.
const int64_t kBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 20;

const int64_t kNumberOfTestEpochs = 1;

// Where to find the MNIST dataset.
const char *kDataFolder = "./training/data/cifar-10-batches-bin";

const char *kCheckPointFolder = "./training/check_points/resent18-cifar10";

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

template <class Module, class Optimizer>
void train(Module &module, Optimizer &optimizer, torch::Device &device)
{
    std::cout << "started the training..." << std::endl;
    // Assume the MNIST dataset is available under `mnist`;
    auto dataset = CIFAR10(kDataFolder)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), kBatchSize);
    std::cout << "Lodded training data set." << std::endl;
    int64_t checkpoint_counter = 0;

    const auto start_time = std::chrono::system_clock::now();

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
    {
        module->train();
        int64_t batch_index = 0;
        for (torch::data::Example<> &batch : *data_loader)
        {
            const auto b_start_time = std::chrono::system_clock::now();
            optimizer.zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = batch.target.to(device);
            torch::Tensor real_output = torch::log_softmax(module->forward(real_images), 1);
            torch::Tensor loss = torch::nll_loss(real_output, real_labels);
            loss.backward();  // calculate the gradiants [the resulting tensor holds the compute graph].
            optimizer.step(); // Apply the calculated grads.
            batch_index++;
            if (batch_index % kLogInterval == 0)
            {
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - b_start_time);
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] | loss: %.4f | Time elapsed: %3ld",
                    epoch,
                    kNumberOfEpochs,
                    batch_index,
                    batches_per_epoch,
                    loss.item<float>(),
                    duration.count());
            }

            if (batch_index % kCheckpointEvery == 0)
            {
                // Checkpoint the model and optimizer state.
                torch::save(module, std::string(kCheckPointFolder) + "/resnet-checkpoint.pt");
                torch::save(optimizer, std::string(kCheckPointFolder) + "/resnet_optimizer-checkpoint.pt");
                std::cout << "\n\rCheck point saved " << ++checkpoint_counter << std::endl;
            }
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start_time);
    auto in_minutes = duration.count() / 60;
    std::cout << "Training complete! Total time took = " << in_minutes << std::endl;
}

template <class Module>
void test(Module &module, torch::Device &device)
{
    auto test_dataset = CIFAR10(
                            kDataFolder, CIFAR10::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
    for (int64_t epoch = 1; epoch <= kNumberOfTestEpochs; ++epoch)
    {
        module->eval();
        double test_loss = 0;
        int32_t correct = 0;
        for (const auto &batch : *test_loader)
        {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = torch::log_softmax(module->forward(data), 1);
            test_loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum).template item<float>();
            correct += output.argmax(1).eq(targets).sum().template item<int64_t>();
        }

        test_loss /= test_dataset_size;
        std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / test_dataset_size);
    }
}

//
//
int main()
{
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);
    //Configs :
    std::vector<uint> n_blocks{2, 2, 2};
    std::vector<uint> n_channels{16, 32, 64};
    std::vector<uint> bottlenecks{};
    uint first_kernel_size = 3;

    ResNetBase base_resnet = ResNetBase(n_blocks, n_channels, bottlenecks, 3, first_kernel_size);
    torch::nn::Linear classification(n_channels[2], 10);
    torch::nn::Sequential module = torch::nn::Sequential(base_resnet, classification);
    module->to(device);
    torch::optim::SGD resnet_optimizer(module->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
    std::cout << "Module is built " << std::endl;
    // std::cout << "=============================================================" << std::endl;
    // std::cout << module << std::endl;
    // std::cout << "=============================================================" << std::endl;
    if (kRestoreFromCheckpoint)
    {
        torch::load(module, std::string(kCheckPointFolder) + "/resnet-checkpoint.pt");
        torch::load(resnet_optimizer, std::string(kCheckPointFolder) + "/resnet_optimizer-checkpoint.pt");
        std::cout << "Loding network weights completed" << std::endl;
    }

    if (kTrain)
        train(module, resnet_optimizer, device);

    if (kTest)
        test(module, device);

    return 0;
}