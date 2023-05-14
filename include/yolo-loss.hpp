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

    auto center_lable = targets.slice(3, 20, 22, 1);
    auto center_prediction = predictions.slice(3, 20, 22, 1);
    auto center_loss = m_mse_loss(center_prediction, center_lable);

    auto wh_lable = targets.slice(3, 22, 24, 1);
    auto wh_prediction = predictions.slice(3, 22, 24, 1);
    auto wh_loss = m_mse_loss(wh_prediction, wh_lable);

    auto class_lable = targets.slice(3, 0, 20, 1);
    auto class_prediction = predictions.slice(3, 0, 20, 1);
    auto class_loss = m_cross_entropy_loss(class_prediction, class_lable);

    return object_loss + center_loss + wh_loss + class_loss;
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
