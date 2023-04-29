// Mimicing
// https://github.com/pytorch/pytorch/blob/abb215e22952ae44b764e501d3552bf219ceb95b/torch/csrc/api/include/torch/data/datasets/mnist.h
// https://pytorch.org/cppdocs/api/classtorch_1_1data_1_1datasets_1_1_m_n_i_s_t.html#class-mnist

#include <cstddef>
#include <string>
#include <vector>
#include "torch/torch.h"

class YOLODataset : public torch::data::datasets::Dataset<YOLODataset>
{
 public:
  // The mode in which the dataset is loaded.
  enum class Mode
  {
    kTrain,
    kTest
  };

  explicit YOLODataset(const std::string& root, Mode mode = Mode::kTrain, int s=7, int64_t img_size=416);

  // https://pytorch.org/cppdocs/api/structtorch_1_1data_1_1_example.html#structtorch_1_1data_1_1_example
  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override;

  bool is_train() const noexcept;

 private:
  std::vector<std::string> m_image_column, m_targets_column;
  Mode m_mode;
  int m_cells_s, m_num_achors;

  int64_t m_image_size;

  std::string train_csv{"train.csv"}, test_csv{"test.csv"}, names_file_name{"class-names.csv"};
  std::string m_image_path, m_target_path;
  std::vector<std::string>class_names{};
  size_t m_num_class;
};