#include "resnet.hpp"
#include "cifar10.hpp"
// #include "resnet_other.hpp"
#include <vector>
#include <chrono>

// The batch size for training.
const int64_t kBatchSize = 128;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 60;

const int64_t kNumberOfTestEpochs = 10;

// Where to find the MNIST dataset.
const char *kDataFolder = "./training/data/cifar-10-batches-bin";

const char *kCheckPointFolder = "./training/check_points/resent18-cifar10";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 500;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = true;

const bool kTest = true;

const int64_t kTestBatchSize = 128;

const bool kTrain = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

//
//
#include "helper.hpp"
int main()
{
    torch::manual_seed(0);
    const float learning_rate{1e-1};
    const float momentum{0.9};
    const float weight_decay{1e-4};

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
    std::vector<uint> n_channels{64, 128, 256};
    std::vector<uint> bottlenecks{};
    uint first_kernel_size = 3;

    ResNetBase base_resnet = ResNetBase(n_blocks, n_channels, bottlenecks, 3, first_kernel_size);
    torch::nn::Sequential module =
        torch::nn::Sequential(base_resnet,
                              torch::nn::Flatten(),
                              torch::nn::Linear(torch::nn::LinearOptions(n_channels[n_channels.size() - 1], 10)),
                              torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1)));
    module->to(device);
    torch::optim::SGD resnet_optimizer(module->parameters(),
                                       torch::optim::SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
    std::cout << "Module is built " << std::endl;
    std::cout << "=============================================================" << std::endl;
    std::cout << module << std::endl;
    std::cout << "=============================================================" << std::endl;
    if (kRestoreFromCheckpoint)
    {
        torch::load(module, std::string(kCheckPointFolder) + "/resnet-checkpoint.pt");
        torch::load(resnet_optimizer, std::string(kCheckPointFolder) + "/resnet_optimizer-checkpoint.pt");
        std::cout << "Loding network weights completed" << std::endl;
    }

    if (kTrain)
    {
        auto train_data = CIFAR10(kDataFolder, CIFAR10::Mode::kTrain)
                              .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                        {0.2023, 0.1994, 0.2010}))
                              .map(torch::data::transforms::Stack<>());
        train(module, resnet_optimizer, device, train_data);
        torch::save(module, std::string(kCheckPointFolder) + "/resnet-checkpoint.pt");
        torch::save(resnet_optimizer, std::string(kCheckPointFolder) + "/resnet_optimizer-checkpoint.pt");
    }

    if (kTest)
    {
        auto test_data = CIFAR10(kDataFolder, CIFAR10::Mode::kTest)
                             .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                       {0.2023, 0.1994, 0.2010}))
                             .map(torch::data::transforms::Stack<>());
        test(module, device, test_data);
    }

    return 0;
}