#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

int main()
{
  Mat frame;
  VideoCapture cap;
  cap.open(0);
  if (!cap.isOpened())
  {
    std::cerr << "Unable to open the camera" << std::endl;
    return -1;
  }

  std::cout << "Start grabing" << std::endl << "Press any key to quit" << std::endl;
  for (;;)
  {
    cap.read(frame);
    if (frame.empty())
    {
      std::cout << "empty frame grabbed" << std::endl;
      break;
    }
    imshow("Camera", frame);
    if (waitKey(5) >= 0)
      break;
  }
  return 0;
}