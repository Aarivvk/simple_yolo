#ifndef C98AFE68_11C4_4686_B6ED_169530FA0165
#define C98AFE68_11C4_4686_B6ED_169530FA0165
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/loss.h>
#include <torch/serialize/input-archive.h>

#include <ostream>
class YOLOLoss
{
 public:
  torch::Tensor compute_loss(torch::Tensor predictions, torch::Tensor targets)
  {
    torch::Tensor grand_loss;

    auto class_labels = targets.slice(3, torch::indexing::None, 20, 1);
    auto class_predictions = predictions.slice(3, torch::indexing::None, 20, 1);
    auto class_loss = m_cross_entropy_loss(class_predictions, class_labels);

    auto box_labels = targets.slice(3, 21, 26, 1);
    auto box_predictions = predictions.slice(3, 21, 26, 1);
    auto box_loss = m_mse_loss(box_predictions, box_labels);

    grand_loss = class_loss * 3 + box_loss * 10;

   return grand_loss;
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
};

#endif /* C98AFE68_11C4_4686_B6ED_169530FA0165 */
