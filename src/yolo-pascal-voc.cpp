#include "yolo-dataset.hpp"

int main()
{
  std::string data_set_root{ "training/data/PASCAL_VOC" };
  YOLODataset y_data_set{ data_set_root,  YOLODataset::Mode::kTrain};

  return 0;
}