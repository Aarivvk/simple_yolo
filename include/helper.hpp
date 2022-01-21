torch::nn::Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
                    int64_t dilation = 1, bool bias = false)
{
    torch::nn::Conv2dOptions conv_options =
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
            .stride(stride)
            .padding(padding)
            .bias(bias)
            .groups(groups)
            .dilation(dilation);

    return conv_options;
}

torch::nn::Conv2dOptions create_conv3x3_options(int64_t in_planes,
                                                int64_t out_planes,
                                                int64_t stride = 1,
                                                int64_t groups = 1,
                                                int64_t dilation = 1)
{
    torch::nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes, /*kerner_size = */ 3, stride,
        /*padding = */ dilation, groups, /*dilation = */ dilation, false);
    return conv_options;
}

torch::nn::Conv2dOptions create_conv1x1_options(int64_t in_planes,
                                                int64_t out_planes,
                                                int64_t stride = 1)
{
    torch::nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes,
        /*kerner_size = */ 1, stride,
        /*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false);
    return conv_options;
}

template <class Module, class Optimizer, class Device, class DataSet>
void train(Module &module, Optimizer &optimizer, Device &device, DataSet data_set)
{
    std::cout << "started the training..." << std::endl;
    // Assume the MNIST dataset is available under `mnist`;
    auto dataset = data_set;
    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), kBatchSize);
    std::cout << "Lodded training data set." << std::endl;
    int64_t checkpoint_counter = 0;

    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
    {
        module->train();
        int64_t batch_index = 0;
        for (torch::data::Example<> &batch : *data_loader)
        {
            optimizer.zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = batch.target.to(device);
            torch::Tensor real_output = module->forward(real_images);
            torch::Tensor loss = torch::nll_loss(real_output, real_labels);
            loss.backward();  // calculate the gradiants [the resulting tensor holds the compute graph].
            optimizer.step(); // Apply the calculated grads.
            batch_index++;
            if (batch_index % kLogInterval == 0)
            {
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] | loss: %.4f",
                    epoch,
                    kNumberOfEpochs,
                    batch_index,
                    batches_per_epoch,
                    loss.item<float>());
            }
        }
    }
    std::cout << " Training complete!" << std::endl;
}

template <class Module, class Device, class DataSet>
void test(Module &module, Device &device, DataSet data_set)
{
    std::cout << "Started Testing" << std::endl;
    auto test_dataset = data_set;
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
    std::cout << "Loading Test data is complete" << std::endl;
    for (int64_t epoch = 1; epoch <= kNumberOfTestEpochs; ++epoch)
    {
        module->eval();
        double test_loss = 0;
        int32_t correct = 0;
        for (const auto &batch : *test_loader)
        {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto output = module->forward(data);
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