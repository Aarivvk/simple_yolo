#include "yolo-dataset.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/rapidcsv.h"

YOLODataset::YOLODataset(const std::string& root, Mode mode, int s, int64_t img_size)
    : m_image_path{ root + "/" + "images" },
      m_target_path{ root + "/" + "labels" },
      m_mode(mode),
      m_cells_s(s),
      m_image_size(img_size)
{
  // read class names from class-names.csv
  std::string names_file_path = root + "/" + names_file_name;
  rapidcsv::Document doc_names(names_file_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(','));
  class_names = doc_names.GetColumn<std::string>(0);
  m_num_class = class_names.size();

  // read the CSV file, train.csv, test.csv
  std::string file_name{};
  if (is_train())
  {
    file_name = root + "/" + train_csv;
  }
  else
  {
    file_name = root + "/" + test_csv;
  }
  rapidcsv::Document doc_csv(file_name, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(','));
  m_image_column = doc_csv.GetColumn<std::string>(0);
  m_targets_column = doc_csv.GetColumn<std::string>(1);
}

torch::data::Example<> YOLODataset::get(size_t index)
{
  std::string img_path = m_image_path + "/" + m_image_column.at(index);
  std::string trgs_path = m_target_path + "/" + m_targets_column.at(index);

  cv::Mat image = cv::imread(img_path);
  auto original_width = image.cols;
  auto original_height = image.rows;
  cv::resize(image, image, cv::Size{ static_cast<int>(m_image_size), static_cast<int>(m_image_size) });
  auto img_tensor = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, torch::kByte).clone();
  img_tensor = img_tensor.permute({ 2, 0, 1 });
  img_tensor = img_tensor.to(torch::kF32);

  rapidcsv::Document targets_docs(trgs_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(' '));

  // Loading all target to variable
  std::vector<uint64_t> class_vec = targets_docs.GetColumn<uint64_t>(0);
  std::vector<float> x_vec = targets_docs.GetColumn<float>(1);
  std::vector<float> y_vec = targets_docs.GetColumn<float>(2);
  std::vector<float> w_vec = targets_docs.GetColumn<float>(3);
  std::vector<float> h_vec = targets_docs.GetColumn<float>(4);
  size_t num_marks = targets_docs.GetRowCount();
  torch::Tensor tens_target = torch::zeros({ m_cells_s, m_cells_s, static_cast<long long>((m_num_class + 5)) });
  for (size_t j = 0; j < num_marks; j++)
  {
    
    auto x = x_vec.at(j);
    auto y = y_vec.at(j);
    auto w = w_vec.at(j);
    auto h = h_vec.at(j);

    int8_t class_id = class_vec.at(j);
    // std::string class_name = class_names.at(class_id);

    // Calculate the targets cell number.
    uint cell_i, cell_j;
    cell_i = y * m_cells_s;
    cell_j = x * m_cells_s;

    // load targets [class, x, y, w, h, obj]
    tens_target[cell_i][cell_j][class_id - 1] = 1;
    tens_target[cell_i][cell_j][m_num_class + 0] = x;
    tens_target[cell_i][cell_j][m_num_class + 1] = y;
    tens_target[cell_i][cell_j][m_num_class + 2] = w;
    tens_target[cell_i][cell_j][m_num_class + 3] = h;
    tens_target[cell_i][cell_j][m_num_class + 4] = 1;
  }
  return { img_tensor, tens_target };
}

torch::optional<size_t> YOLODataset::size() const
{
  return m_image_column.size();
}

bool YOLODataset::is_train() const noexcept
{
  return m_mode == Mode::kTrain;
}
