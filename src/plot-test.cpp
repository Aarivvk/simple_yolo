#include "utils/matplot.hpp"


int main()
{
    DataPloter data_ploter;

    for (size_t i = 0; i < 100; i++)
    {
        data_ploter.add_data_train(i*1.5, i, i*0.5);
        data_ploter.add_data_validation(i*2, i, i*1.5);
        data_ploter.show_plot();
    }
    data_ploter.show_plot();
    std::cout << "Waiting for user input" << std::endl;
    int input;

    std::cin >> input;
    


}