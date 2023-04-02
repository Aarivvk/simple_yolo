#ifndef FC428CF0_7001_404D_8789_3DD8AA765EA4
#define FC428CF0_7001_404D_8789_3DD8AA765EA4

#include <opencv2/core/hal/interface.h>

#include <cstddef>
#include <string>
namespace VOC
{
  class InputData
  {
  };

  class Target
  {
   public:
    enum class Type
    {
      MID_POINT,
      TOP_LEFT
    };
    Target()
    {
    }
    Target(float x, float y, float w, float h, size_t c, std::string c_name = "None", Type t = Type::MID_POINT)
        : x{ x },
          y{ y },
          w{ w },
          h{ h },
          c{ c },
          c_name{ c_name },
          type(t)
    {
    }
    float x{}, y{}, w{}, h{};
    size_t c{};
    std::string c_name{};
    Type type{ Type::MID_POINT };
  };
}  // namespace YOLO

#endif /* FC428CF0_7001_404D_8789_3DD8AA765EA4 */
