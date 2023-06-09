#include "yolo-dataset.hpp"

#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "toml++/toml.h"
#include "utils/rapidcsv.h"

YOLODataset::YOLODataset(Mode mode, toml::node_view<toml::node> config) : YOLODataset{}
{
  std::filesystem::path root = config["training_data_root"].value<std::string>().value();
  check_file(root);
  m_image_path = root / config["image_directory_name"].value<std::string>().value();
  check_file(m_image_path);
  m_target_path = root / config["lable_directory_name"].value<std::string>().value();
  check_file(m_target_path);
  m_mode = mode;
  m_cells_s = config["number_of_detections"].value<int>().value();
  m_image_width = config["image_width"].value<int>().value();
  m_image_height = config["image_height"].value<int>().value();

  // read class names from class-names.csv
  std::filesystem::path names_file_path = root / config["class_names"].value<std::string>().value();
  check_file(names_file_path);
  rapidcsv::Document doc_names(names_file_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(','));
  m_class_names = doc_names.GetColumn<std::string>(0);
  m_num_class = m_class_names.size();

  // read the CSV file, train.csv, test.csv
  std::filesystem::path file_name{};
  if (is_train())
  {
    file_name = root / config["train_file_name"].value<std::string>().value();
  }
  else
  {
    file_name = root / config["test_file_name"].value<std::string>().value();
  }
  check_file(file_name);
  rapidcsv::Document doc_csv(file_name, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(','));
  m_image_column = doc_csv.GetColumn<std::string>(0);
  m_targets_column = doc_csv.GetColumn<std::string>(1);
}

void YOLODataset::check_file(std::filesystem::path root)
{
  bool return_code = std::filesystem::exists(root);
  if (!return_code)
  {
    throw std::invalid_argument("file/directory does not exists! " + root.string());
  }
}

torch::data::Example<> YOLODataset::get(size_t index)
{
  std::filesystem::path img_path = m_image_path / m_image_column.at(index);
  std::filesystem::path trgs_path = m_target_path / m_targets_column.at(index);
  check_file(img_path);
  check_file(trgs_path);
  cv::Mat image = cv::imread(img_path);
  cv::Size img_size;
  img_size.width = static_cast<int>(m_image_width);
  img_size.height = static_cast<int>(m_image_height);

  cv::resize(image, image, img_size);
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
    cell_i = x * m_cells_s;
    cell_j = y * m_cells_s;

    // load targets [class, x, y, w, h, obj]
    tens_target[cell_i][cell_j][class_id] = 1;
    tens_target[cell_i][cell_j][m_num_class + 0] = x;
    tens_target[cell_i][cell_j][m_num_class + 1] = y;
    tens_target[cell_i][cell_j][m_num_class + 2] = w;
    tens_target[cell_i][cell_j][m_num_class + 3] = h;
    tens_target[cell_i][cell_j][m_num_class + 4] = 1;
  }

  img_tensor = (img_tensor / 255);
  img_tensor = (img_tensor - img_tensor.mean())/ img_tensor.std();

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
