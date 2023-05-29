#ifndef C98AFE68_11C4_4686_B6ED_169530FA0165
#define C98AFE68_11C4_4686_B6ED_169530FA0165
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/loss.h>
#include <torch/serialize/input-archive.h>

#include <ostream>
class YOLOLossImpl : public torch::nn::Module
{
 public:
  torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
  {
    auto bojectness_labels = targets.slice(3, 24, 25, 1);
    auto bojectness_predictions = predictions.slice(3, 24, 25, 1);
    auto object_loss = m_bce(bojectness_predictions, bojectness_labels);

    auto center_x_lable = targets.slice(3, 20, 21, 1);
    auto center_x_prediction = predictions.slice(3, 20, 21, 1).sigmoid();
    auto center_x_loss = m_mse_loss(center_x_prediction, center_x_lable);

    auto center_y_lable = targets.slice(3, 21, 22, 1);
    auto center_y_prediction = predictions.slice(3, 21, 22, 1).sigmoid();
    auto center_y_loss = m_mse_loss(center_y_prediction, center_y_lable);

    auto w_lable = targets.slice(3, 22, 23, 1);
    auto w_prediction = predictions.slice(3, 22, 23, 1).sigmoid();
    auto w_loss = m_mse_loss(w_prediction, w_lable);

    auto h_lable = targets.slice(3, 23, 24, 1);
    auto h_prediction = predictions.slice(3, 23, 24, 1).sigmoid();
    auto h_loss = m_mse_loss(h_prediction, h_lable);

    auto class_lable = targets.slice(3, 0, 20, 1).permute({ 0, 3, 1, 2 });
    // torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(1));
    auto class_prediction = predictions.slice(3, 0, 20, 1).permute({ 0, 3, 1, 2 });
    auto class_loss = m_cross_entropy_loss(class_prediction, class_lable);

    // auto boxes_lable = targets.slice(3, 20, 24, 1);
    // auto boxes_predition = predictions.slice(3, 20, 24, 1).sigmoid();
    // auto iou = IOU(boxes_predition, boxes_lable);

    // std::cout << "object_loss " << object_loss << std::endl;
    // std::cout << "center_loss " << center_loss << std::endl;
    // std::cout << "wh_loss " << wh_loss << std::endl;
    // std::cout << "class_loss " << center_loss << std::endl << std::endl;

    return object_loss + center_x_loss + center_y_loss + w_loss + h_loss + class_loss;
  }

  double accuracy(torch::Tensor predictions, torch::Tensor targets)
  {
    uint64 true_positive = 0, false_positive = 0;
    double Precision{};
    auto classes = predictions.slice(2, torch::indexing::None, 20, 1);
    torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(1));
    auto classess_flaten = classes.flatten(1, 2);

    auto objectness = predictions.slice(2, 24, torch::indexing::None, 1);
    auto objectness_flaten = objectness.flatten(1, 2).squeeze();
    classess_flaten = softmax(classess_flaten);
    objectness_flaten = objectness_flaten.sigmoid();
    size_t batch_size = predictions.size(0);
    size_t number_of_detections = objectness_flaten.size(1);


    std::vector<int> selected_index{};
    for (size_t i = 0; i < batch_size; i++)
    {
      for(size_t j=0; j<number_of_detections; j++ )
      {
        auto class_indexe = classess_flaten[i][j].argmax().item<int>();
        auto calss_prob = classess_flaten[i][j][class_indexe].item<double>();
        auto objectness_prob = objectness_flaten[i][j].item<double>();
      if (objectness_prob > 0.5)
      {
        std::cout << "Selecting the index " << i << " with objectness_prob " << objectness_prob << " calss_prob "
                  << calss_prob << " class_index " << class_indexe << std::endl;
        selected_index.push_back(i);
      }

      }

    }

    return Precision;
  }

 private:
  /**
  @brief For binary classification error.
  */
  torch::nn::CrossEntropyLoss m_cross_entropy_loss;

  /**
  @brief For regrestion error, for continuos values like x,y,w,h.
  */
  torch::nn::MSELoss m_mse_loss;

  /**
  @brief For objectness loss.
  */
  torch::nn::BCEWithLogitsLoss m_bce;

  torch::Tensor IOU(torch::Tensor box1, torch::Tensor box2)
  {
    auto x1c = box1.slice(1, 0, 1, 1);
    auto y1c = box1.slice(1, 1, 2, 1);
    auto w1 = box1.slice(1, 2, 3, 1);
    auto h1 = box1.slice(1, 3, 4, 1);
    auto x11 = x1c - w1 / 2;
    auto y11 = y1c - h1 / 2;
    auto x12 = x1c + w1 / 2;
    auto y12 = y1c + h1 / 2;

    auto areas1 = w1 * h1;

    auto x2c = box2.slice(1, 0, 1, 1);
    auto y2c = box2.slice(1, 1, 2, 1);
    auto w2 = box2.slice(1, 2, 3, 1);
    auto h2 = box2.slice(1, 3, 4, 1);
    auto x21 = x2c - w2 / 2;
    auto y21 = y2c - h2 / 2;
    auto x22 = x2c + w2 / 2;
    auto y22 = y2c + h2 / 2;

    auto areas2 = w2 * h2;

    auto left_top = torch::max(torch::cat({ x11, y11 }, 1), torch::cat({ x21, y11 }, 1));
    auto right_bottom = torch::max(torch::cat({ x12, y12 }, 1), torch::cat({ x22, y22 }, 1));
    auto width_height_intersection = (right_bottom - left_top).clamp(0);
    auto intersection = width_height_intersection.slice(1, 0, 1, 1) * width_height_intersection.slice(1, 1, 2, 1);
    auto u_n_i_o_n = areas1 + areas2 - intersection;
    auto iou = intersection / u_n_i_o_n;
    return iou;
  }
};

TORCH_MODULE(YOLOLoss);
#endif /* C98AFE68_11C4_4686_B6ED_169530FA0165 */
