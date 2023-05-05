// Mimicing
// https://github.com/pytorch/pytorch/blob/abb215e22952ae44b764e501d3552bf219ceb95b/torch/csrc/api/include/torch/data/datasets/mnist.h
// https://pytorch.org/cppdocs/api/classtorch_1_1data_1_1datasets_1_1_m_n_i_s_t.html#class-mnist

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "toml++/toml.h"
#include "torch/torch.h"

class YOLODataset : public torch::data::datasets::Dataset<YOLODataset>
{
 public:
  // The mode in which the dataset is loaded.
  enum class Mode
  {
    kTrain,
    kTest,
    kValidation
  };
  explicit YOLODataset(Mode mode, toml::node_view<toml::node> config);

  // https://pytorch.org/cppdocs/api/structtorch_1_1data_1_1_example.html#structtorch_1_1data_1_1_example
  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override;

  bool is_train() const noexcept;

 private:
  YOLODataset() = default;
  void check_file(std::filesystem::path);
  std::filesystem::path m_image_path{}, m_target_path{};

  Mode m_mode{};
  int m_cells_s{}, m_num_achors{};

  int64_t m_image_width{}, m_image_height{};
  std::vector<std::string> m_image_column{}, m_targets_column{};

  std::vector<std::string> m_class_names{};
  size_t m_num_class{};
};