#include <torch/torch.h>
#include <cmath>
#include <vector>

struct ShortcutProjectionImpl : torch::nn::Module
{
    ShortcutProjectionImpl(uint in_channels, uint out_channels, uint stride) : conv1(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 1).bias(false).stride(stride)),
                                                                               batch_norm1(out_channels)
    {
        register_module("conv1", conv1);
        register_module("batch_norm1", batch_norm1);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return batch_norm1(conv1(x));
    }

    torch::nn::ConvTranspose2d conv1;
    torch::nn::BatchNorm2d batch_norm1;
};

TORCH_MODULE(ShortcutProjection);

struct ResidualBlockImpl : torch::nn::Module
{
    ResidualBlockImpl(uint in_channels, uint out_channels, uint stride) : conv1(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)), batch_norm1(out_channels), conv2(torch::nn::ConvTranspose2dOptions(out_channels, out_channels, 3).stride(1).padding(1)), batch_norm2(out_channels)
    {
        register_module("conv1", conv1);
        register_module("batch_norm1", batch_norm1);
        register_module("conv2", conv2);
        register_module("batch_norm2", batch_norm2);
        if (stride != 1 || in_channels != out_channels)
        {
            sshort = ShortcutProjection(in_channels, out_channels, stride);
            register_module("short_cut", sshort);
        }
        else
        {
            sshort_id = torch::nn::Identity();
            register_module("short_id", sshort_id);
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // std::cout << "BB Predicting step 0" << x.sizes() << std::endl;
        torch::Tensor short_mat;
        if (!sshort.is_empty())
        {
            short_mat = sshort->forward(x);
        }
        else
        {
            short_mat = sshort_id(x);
        }
        x = act1(batch_norm1(conv1(x)));
        // std::cout << "BB Predicting step 1" << x.sizes() << std::endl;
        x = batch_norm2(conv2(x));
        // std::cout << "BB Predicting step 2" << x.sizes() << std::endl;
        x = act2(x + short_mat);
        // std::cout << "BB Predicting step complete" << x.sizes() << std::endl;

        return x;
    }

    torch::nn::ConvTranspose2d conv1, conv2;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2;
    ShortcutProjection sshort = nullptr;
    torch::nn::Identity sshort_id = nullptr;
    torch::nn::ReLU act1, act2;
};

TORCH_MODULE(ResidualBlock);

struct BottleneckResidualBlockImpl : torch::nn::Module
{
    BottleneckResidualBlockImpl(uint in_channels, uint bottleneck_channels, uint out_channels, uint stride) : conv1(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 1).stride(1)),
                                                                                                              batch_norm1(out_channels),
                                                                                                              conv2(torch::nn::ConvTranspose2dOptions(out_channels, out_channels, 3).stride(stride).padding(1)),
                                                                                                              batch_norm2(out_channels),
                                                                                                              conv3(torch::nn::ConvTranspose2dOptions(out_channels, bottleneck_channels, 1).stride(1)),
                                                                                                              batch_norm3(bottleneck_channels)
    {
        register_module("conv1", conv1);
        register_module("batch_norm1", batch_norm1);
        register_module("conv2", conv2);
        register_module("batch_norm2", batch_norm2);
        register_module("conv3", conv3);
        register_module("batch_norm3", batch_norm3);
        if (stride != 1 || in_channels != out_channels)
        {
            sshort = ShortcutProjection(in_channels, bottleneck_channels, stride);
            register_module("short_cut", sshort);
        }
        else
        {
            sshort_id = torch::nn::Identity();
            register_module("short_id", sshort_id);
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // std::cout << "BN Predicting step 0" << x.sizes() << std::endl;

        torch::Tensor short_mat;
        if (!sshort.is_empty())
        {
            short_mat = sshort->forward(x);
        }
        else
        {
            short_mat = sshort_id(x);
        }
        x = act1(batch_norm1(conv1(x)));
        // std::cout << "BN Predicting step 1" << x.sizes() << std::endl;
        x = act2(batch_norm2(conv2(x)));
        // std::cout << "BN Predicting step 2" << x.sizes() << std::endl;
        x = batch_norm3(conv3(x));
        // std::cout << "BN Predicting step 3" << x.sizes() << std::endl;
        x = act3(x + short_mat);
        // std::cout << "BN Predicting step complete" << x.sizes() << std::endl;
        return x;
    }

    torch::nn::ConvTranspose2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
    torch::nn::ReLU act1, act2, act3;
    ShortcutProjection sshort = nullptr;
    torch::nn::Identity sshort_id = nullptr;
};

TORCH_MODULE(BottleneckResidualBlock);

struct ResNetBaseImpl : torch::nn::Module
{
    ResNetBaseImpl(std::vector<uint> n_blocks, std::vector<uint> n_channels, std::vector<uint> bottlenecks, uint img_channels = 3, uint first_kernel_size = 7) : conv1(torch::nn::ConvTranspose2dOptions(img_channels, n_channels[0], first_kernel_size).stride(2).padding(std::floor(first_kernel_size / 2))), batch_norm1(n_channels[0])
    {
        register_module("conv1", conv1);
        register_module("batch_norm1", batch_norm1);
        size_t i = 0;
        uint prev_channels = n_channels[0];
        bool is_first_block{true};
        // std::cout << "Number of blocks " << n_channels.size() << std::endl;
        for (uint channels : n_channels)
        {
            // std::cout << "number of layers " << n_blocks[i] << std::endl;
            size_t stride = 0;
            if (is_first_block)
            {
                is_first_block = false;
                stride = 2;
            }
            else
            {
                stride = 1;
            }
            if (0 == bottlenecks.size())
            {
                sequenc_blocks->push_back(ResidualBlock(prev_channels, channels, stride));
            }
            else
            {
                sequenc_blocks->push_back(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels, stride));
            }
            prev_channels = channels;

            ++i;
        }
        register_module("sequenc_blocks", sequenc_blocks);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // std::cout << "Predicting step 0" << x.sizes() << std::endl;
        x = batch_norm1(conv1(x));
        // std::cout << "Predicting step 1" << x.sizes() << std::endl;
        x = sequenc_blocks->forward(x);
        // std::cout << "Predicting step 2" << x.sizes() << std::endl;
        x = x.view({x.sizes()[0], x.sizes()[1], -1});
        // std::cout << "Predicting step 3" << x.sizes() << std::endl;
        x = x.mean(-1);
        // std::cout << "Predicting step complete " << x.sizes() << std::endl;
        return x;
    }

    torch::nn::ConvTranspose2d conv1;
    torch::nn::BatchNorm2d batch_norm1;
    torch::nn::Sequential sequenc_blocks;
};

TORCH_MODULE(ResNetBase);