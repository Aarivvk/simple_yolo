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
    m_figure->quiet_mode(true);
    m_figure->ioff();
    m_figure->title("Yolo plot");
    m_figure->size(2048, 1024);
    matplot::tiledlayout(1, 2);
    m_ax1 = matplot::nexttile();
    matplot::title(m_ax1, "Loss Plot");
    matplot::ylabel(m_ax1, "loss");
    matplot::xlabel(m_ax1, "epochs");
    matplot::yrange(m_ax1, { 0 });

    m_ax2 = matplot::nexttile();
    matplot::title(m_ax2, "Precision Plot");
    matplot::ylabel(m_ax2, "Precision");
    matplot::xlabel(m_ax2, "epochs");
    matplot::yrange(m_ax1, { 0, 1 });
  }
  void add_data_train(double loss, size_t x, double precision = 0)
  {
    m_train_loss_vec.push_back(loss);
    m_train_precision_vec.push_back(precision);
    m_train_x_vec.push_back(x);
  }

  void add_data_validation(double loss, size_t x, double precision = 0)
  {
    m_validation_x_vec.push_back(x);
    m_test_precision_vec.push_back(precision);
    m_validation_loss_vec.push_back(loss);
  }

  bool show_plot()
  {
    plot_data();
    if (m_train_x_vec.size() > 0 || m_validation_x_vec.size() > 0)
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
  std::vector<double> m_train_loss_vec, m_validation_loss_vec, m_train_precision_vec;
  std::vector<double> m_train_x_vec, m_validation_x_vec, m_test_precision_vec;
  matplot::figure_handle m_figure;
  matplot::axes_handle m_ax1, m_ax2;

  void plot_data()
  {
    if (m_train_x_vec.size() > 0)
    {
      auto p_train_loss = matplot::plot(m_ax1, m_train_x_vec, m_train_loss_vec);
      p_train_loss->line_width(2).color("g");
    }

    matplot::hold(matplot::on);

    if (m_validation_x_vec.size() > 0)
    {
      auto p_test_loss = matplot::plot(m_ax1, m_validation_x_vec, m_validation_loss_vec);
      p_test_loss->use_y2();
      p_test_loss->line_width(2).color("b");
    }

    if (m_train_x_vec.size() > 0)
    {
      auto p_train_precision = matplot::plot(m_ax2, m_train_x_vec, m_train_precision_vec);
      p_train_precision->line_width(2).color("g");
    }

    matplot::hold(matplot::on);

    if (m_validation_x_vec.size() > 0)
    {
      auto p_test_precision = matplot::plot(m_ax2, m_validation_x_vec, m_test_precision_vec);
      p_test_precision->use_y2();
      p_test_precision->line_width(2).color("b");
    }
  }
};

#endif /* BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A */
