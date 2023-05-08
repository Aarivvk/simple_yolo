#ifndef BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A
#define BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A
#include <matplot/matplot.h>

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
    m_figure->should_close();
    m_figure->ioff();
    m_ax_h = m_figure->add_axes();
    m_main_loss_line = m_ax_h->loglog(m_los_vec);
    m_main_loss_line->color("red");
  }
  void add_data(double loss)
  {
    m_los_vec.push_back(loss);
    m_main_loss_line->y_data(m_los_vec);
  }

  void show_plot()
  {
    m_figure->draw();
  }

 private:
  std::vector<double> m_los_vec;
  matplot::figure_handle m_figure;
  matplot::axes_handle m_ax_h;
  matplot::line_handle m_main_loss_line;
};

#endif /* BAB5ADEF_3FDC_40A5_A25E_7750FFDCAD4A */
