#include "yolo-dataset.hpp"

#include <ATen/ScalarOps.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <opencv2/core/hal/interface.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>
#include <vector>

#include "bar-loop.hpp"
#include "comm/voc-data.hpp"
#include "open-cv-helper.hpp"
#include "rapidcsv.h"
#include "torch/torch.h"

YOLODataset::YOLODataset(const std::string& root, Mode mode, int s, int b)
    : m_image_path{ root + "/" + "images" },
      m_target_path{ root + "/" + "labels" },
      m_mode(mode),
      m_cells_s(s)
{
  // read class names from class-names.csv
  std::string names_file_path = root + "/" + names_file_name;
  rapidcsv::Document doc_names(names_file_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(','));
  class_names = doc_names.GetColumn<std::string>(0);
  int64_t num_classes = class_names.size();

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
  std::vector<std::string> image_column = doc_csv.GetColumn<std::string>(0);
  std::vector<std::string> targets_column = doc_csv.GetColumn<std::string>(1);

  m_size = image_column.size();
  m_images = torch::zeros({ m_size, 3, 416, 416 });
  m_targets = torch::zeros({ m_size, m_cells_s, m_cells_s, num_classes + 4 });

  std::cout << "Loding data from hard disk" << std::endl;
  for (size_t i : vk::intToRange(m_size))
  {
    std::string img_path = m_image_path + "/" + image_column.at(i);
    std::string trgs_path = m_target_path + "/" + targets_column.at(i);

    cv::Mat image = cv::imread(img_path);
    cv::resize(image, image, cv::Size{ 416, 416 });
    int64_t width = image.size().width;
    int64_t height = image.size().height;
    int64_t channel = image.channels();
    auto img_tensor =
        torch::tensor(at::ArrayRef<uchar>(image.data, width * height * channel)).view({ channel, height, width });
    // load images into m_images
    m_images[i] = img_tensor;

    rapidcsv::Document targets_docs(trgs_path, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(' '));

    // Loading all target to variable
    std::vector<uint64_t> class_vec = targets_docs.GetColumn<uint64_t>(0);
    std::vector<float> x_vec = targets_docs.GetColumn<float>(1);
    std::vector<float> y_vec = targets_docs.GetColumn<float>(2);
    std::vector<float> w_vec = targets_docs.GetColumn<float>(3);
    std::vector<float> h_vec = targets_docs.GetColumn<float>(4);
    size_t num_marks = targets_docs.GetRowCount();
    for (size_t j = 0; j < num_marks; j++)
    {
      auto x = x_vec.at(j) * width;
      auto y = y_vec.at(j) * height;
      auto w = w_vec.at(j) * width;
      auto h = h_vec.at(j) * height;
      int8_t class_id = class_vec.at(j);
      std::string class_name = class_names.at(class_id);

      // Calculate the the row and column number in check target recides.
      uint cell_i, cell_j;
      cell_i = floor(x / (m_cells_s * m_cells_s));
      cell_j = floor(y / (m_cells_s * m_cells_s));
      // trim the cell row and column
      cell_i = cell_i >= m_cells_s ? m_cells_s - 1 : (cell_i <= 0 ? 0 : cell_i);
      cell_j = cell_j >= m_cells_s ? m_cells_s - 1 : (cell_j <= 0 ? 0 : cell_j);

      // load targets [class, x, y, w, h]
      m_targets[i][cell_i][cell_j][class_id - 1] = 1;
      m_targets[i][cell_i][cell_j][20] = x;
      m_targets[i][cell_i][cell_j][21] = y;
      m_targets[i][cell_i][cell_j][22] = w;
      m_targets[i][cell_i][cell_j][23] = h;

      // VOC::Target target{ x, y, w, h, static_cast<size_t>(class_id), class_name };
      // mark_target(image, target);
    }

    // mark_cells(image, m_cells_s);
    // display_image(image);
    // while (is_quit())
    // {
    //   exit(2);
    // }
  }

  std::cout << "Done loading data set " << std::endl;
}

torch::data::Example<> YOLODataset::get(size_t index)
{
  return { m_images[index], m_targets[index] };
}

torch::optional<size_t> YOLODataset::size() const
{
  return m_size;
}

bool YOLODataset::is_train() const noexcept
{
  return m_mode == Mode::kTrain;
}

// Returns all images stacked into a single tensor.
const torch::Tensor& YOLODataset::images() const
{
  return m_images;
}

const torch::Tensor& YOLODataset::targets() const
{
  return m_targets;
}
