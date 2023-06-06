#ifndef C98AFE68_11C4_4686_B6ED_169530FA0165
#define C98AFE68_11C4_4686_B6ED_169530FA0165
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/loss.h>
#include <torch/serialize/input-archive.h>

#include <ostream>

using torch::indexing::Ellipsis;
using torch::indexing::None;
using torch::indexing::Slice;

class YOLOLossImpl : public torch::nn::Module
{
 public:
  torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
  {
    auto objects_index = targets.index({ Ellipsis, 24 }) == 1;
    auto no_objects_index = targets.index({ Ellipsis, 24 }) == 0;
    // std::cout << "objects_index " << objects_index.sizes() << std::endl;
    // std::cout << "no_objects_index " << no_objects_index.sizes() << std::endl;
    // objects_index [15, 7, 7]
    // no_objects_index [15, 7, 7]

    auto objects_label = targets.index({ objects_index });
    auto objects_predictions = predictions.index({ objects_index });
    // std::cout << "objects_label " << objects_label.sizes() << std::endl;
    // std::cout << "objects_predictions " << objects_predictions.sizes() << std::endl;
    // objects_label[33, 25]
    // objects_predictions[33, 25]

    auto object_loss = yolo_loss(objects_label, objects_predictions);

    auto no_objects_label = targets.index({ no_objects_index });
    auto no_objects_predictions = predictions.index({ no_objects_index });
    auto no_object_loss = yolo_loss(no_objects_label, no_objects_predictions);

    // std::cout << "object_loss " << object_loss.item<double>() << std::endl;
    // std::cout << "no_object_loss " << no_object_loss.item<double>() << std::endl;

    return object_loss + no_object_loss;
  }

  torch::Tensor yolo_loss(torch::Tensor& objects_label, torch::Tensor& objects_predictions)
  {
    // Objectness loss
    auto obj_bojectness_labels = objects_label.slice(1, 24, 25, 1);
    auto obj_bojectness_predictions = objects_predictions.slice(1, 24, 25, 1);
    auto object_loss = m_bce(obj_bojectness_predictions, obj_bojectness_labels);

    // center x loss
    auto center_x_lable = objects_label.slice(1, 20, 21, 1);
    auto center_x_prediction = objects_predictions.slice(1, 20, 21, 1).sigmoid();
    auto center_x_loss = m_mse_loss(center_x_prediction, center_x_lable);

    // center y loss
    auto center_y_lable = objects_label.slice(1, 21, 22, 1);
    auto center_y_prediction = objects_predictions.slice(1, 21, 22, 1).sigmoid();
    auto center_y_loss = m_mse_loss(center_y_prediction, center_y_lable);

    // width loss
    auto w_lable = objects_label.slice(1, 22, 23, 1);
    auto w_prediction = objects_predictions.slice(1, 22, 23, 1).sigmoid();
    auto w_loss = m_mse_loss(w_prediction, w_lable);

    // height loss
    auto h_lable = objects_label.slice(1, 23, 24, 1);
    auto h_prediction = objects_predictions.slice(1, 23, 24, 1).sigmoid();
    auto h_loss = m_mse_loss(h_prediction, h_lable);

    // Object class loss
    auto class_lable = objects_label.slice(1, 0, 20, 1);
    auto class_prediction = objects_predictions.slice(1, 0, 20, 1);
    auto class_loss = m_cross_entropy_loss(class_prediction, class_lable);

    return object_loss + center_x_loss + center_y_loss + w_loss + h_loss + class_loss;
  }

  double accuracy(torch::Tensor predictions, torch::Tensor targets)
  {
    torch::NoGradGuard no_grad;
    double true_positive = 0, true_negative = 0, false_positive = 0, false_negative = 0;
    double accuracy{};
    auto classes = predictions.slice(3, None, 20, 1);
    auto classes_lable = targets.slice(3, None, 20, 1);
    torch::nn::Softmax softmax(torch::nn::SoftmaxOptions(2));
    auto classess_flaten = classes.flatten(1, 2);
    auto classess_flaten_label = classes_lable.flatten(1, 2);

    auto objectness = predictions.slice(3, 24, None, 1);
    auto objectness_lable = targets.slice(3, 24, None, 1);
    auto objectness_flaten = objectness.flatten(1, 2);
    auto objectness_lable_flaten = objectness_lable.flatten(1, 2);
    classess_flaten = softmax(classess_flaten);
    classess_flaten_label = softmax(classess_flaten_label);
    objectness_flaten = objectness_flaten;
    size_t batch_size = predictions.size(0);
    size_t number_of_detections = objectness_flaten.size(1);

    for (size_t i = 0; i < batch_size; i++)
    {
      for (size_t j = 0; j < number_of_detections; j++)
      {
        auto class_indexe = classess_flaten[i][j].argmax().item<int>();
        auto calss_prob = classess_flaten[i][j][class_indexe].item<double>();

        auto class_indexe_label = classess_flaten_label[i][j].argmax().item<int>();
        auto calss_prob_lable = classess_flaten_label[i][j][class_indexe].item<double>();

        auto objectness_prob = objectness_flaten[i][j].item<double>();
        auto objectness_lable_prob = objectness_lable_flaten[i][j].item<double>();
        if ((objectness_prob > 0.5 && objectness_lable_prob == 1.00))
        {
          ++true_positive;
        }
        if ((objectness_prob < 0.5 && objectness_lable_prob == 0.00))
        {
          ++true_negative;
        }

        if (objectness_prob > 0.5 && objectness_lable_prob == 0.00)

        {
          ++false_positive;
        }
        if (objectness_prob < 0.5 && objectness_lable_prob == 1.00)
        {
          ++false_negative;
        }
      }
    }

    auto numarator = true_positive + true_negative;
    auto devider = (numarator + false_positive + false_negative);
    if (devider > 0.0)
    {
      accuracy = numarator / devider;
    }

    std::cout << "\e[A\r"
              << "\033[2K"
              << "TP = " << true_positive << " TN = " << true_negative << " FP = " << false_positive
              << " FN = " << false_negative << std::endl
              << std::flush;

    return accuracy;
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
