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
  }
  void add_data_train(double loss, size_t x)
  {
    m_train_loss_vec.push_back(loss);
    m_train_x_vec.push_back(x);
  }

  void add_data_validation(double loss, size_t x)
  {
    m_validation_x_vec.push_back(x);
    m_validation_loss_vec.push_back(loss);
  }

  bool show_plot()
  {
    plot_data();
    m_figure->draw();
    return m_figure->should_close();
  }

  void save_graph(std::string file_name)
  {
    plot_data();
    matplot::save(file_name);
  }

 private:
  std::vector<double> m_train_loss_vec, m_validation_loss_vec;
  std::vector<double> m_train_x_vec, m_validation_x_vec;
  matplot::figure_handle m_figure;

  void plot_data()
  {
    if (m_train_x_vec.size() > 0)
    {
      auto p_train = matplot::loglog(m_train_x_vec, m_train_loss_vec, "g");
      p_train->line_width(2);
    }

    if (m_validation_x_vec.size() > 0)
    {
      auto p_test = matplot::loglog(m_validation_x_vec, m_validation_loss_vec, "b");
      p_test->line_width(2);
    }
  }
};

#endif /* BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A */
