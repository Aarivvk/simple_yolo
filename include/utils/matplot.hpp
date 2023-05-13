#ifndef BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A
#define BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A
#include <matplot/matplot.h>
#include <sys/_types/_size_t.h>

#include <unordered_map>
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
    matplot::plot(m_train_x_vec, m_train_loss_vec, "g", m_validation_x_vec, m_validation_loss_vec, "b--o");
    m_figure->draw();
    return m_figure->should_close();
  }

  void save_graph(std::string file_name)
  {
    m_figure->save(file_name);
  }

 private:
  std::vector<double> m_train_loss_vec, m_validation_loss_vec;
  std::vector<double> m_train_x_vec, m_validation_x_vec;
  matplot::figure_handle m_figure;
};

#endif /* BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A */
