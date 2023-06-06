#ifndef BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A
#define BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A

#include <matplot/matplot.h>

#include <vector>

class DataPloter
{
 public:
  DataPloter()
  {
    m_figure = matplot::figure();
    m_figure->reactive_mode(false);
    m_figure->title("Yolo plot");
    m_figure->size(2048, 1024);

    m_ax1 = m_figure->add_subplot(1, 2, 0);
    m_ax1->title("Loss Plot");
    m_ax1->ylabel("loss");
    m_ax1->xlabel("epochs");
    m_ax1->legend({ "Train", "Test" });

    m_ax2 = m_figure->add_subplot(1, 2, 1);
    m_ax2->title("Precision Plot");
    m_ax2->ylabel("Precision");
    m_ax2->xlabel("epochs");
    m_ax2->legend({ "Train", "Test" });
  }
  void add_data_train(double loss, size_t x, double precision = 0)
  {
    m_train_loss_vec.push_back(loss);
    m_train_precision_vec.push_back(precision);
    m_train_x_vec.push_back(x);
  }

  void add_data_validation(double loss, size_t x, double precision = 0)
  {
    m_test_x_vec.push_back(x);
    m_test_precision_vec.push_back(precision);
    m_test_loss_vec.push_back(loss);
  }

  bool show_plot()
  {
    plot_data();
    if (m_train_x_vec.size() > 0 || m_test_x_vec.size() > 0)
    {
      m_figure->draw();
    }
    return m_figure->should_close();
  }

  void save_graph(std::string file_name)
  {
    plot_data();
    matplot::save(file_name);
  }

 private:
  std::vector<double> m_train_loss_vec, m_test_loss_vec, m_train_precision_vec;
  std::vector<double> m_train_x_vec, m_test_x_vec, m_test_precision_vec;
  matplot::figure_handle m_figure;
  matplot::axes_handle m_ax1, m_ax2;

  void plot_data()
  {
    auto p_train_loss = m_ax1->plot(m_train_x_vec, m_train_loss_vec);
    p_train_loss->line_width(2).color("g");
    m_ax1->hold(matplot::on);

    auto p_test_loss = m_ax1->plot(m_test_x_vec, m_test_loss_vec);
    p_test_loss->use_y2(true);
    p_test_loss->line_width(2).color("b");
    m_ax1->hold(matplot::off);

    auto p_train_precision = m_ax2->plot(m_train_x_vec, m_train_precision_vec);
    p_train_precision->line_width(2).color("g");
    m_ax2->hold(matplot::on);

    auto p_test_precision = m_ax2->plot(m_test_x_vec, m_test_precision_vec);
    p_test_precision->use_y2(true);
    p_test_precision->line_width(2).color("b");
    m_ax2->hold(matplot::off);
  }
};

#endif /* BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A */
