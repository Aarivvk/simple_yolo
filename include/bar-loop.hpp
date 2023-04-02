#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "indicators.hpp"

namespace vk
{
  using namespace indicators;

  inline std::shared_ptr<ProgressBar> build_bar(std::string preFix = "")
  {
    std::shared_ptr<ProgressBar> bar = std::make_shared<ProgressBar>(
        option::BarWidth{ 100 },
        option::Start{ "[" },
        option::Fill{ "█" },
        option::Lead{ ">" },
        option::Remainder{ "-" },
        option::End{ "]" },
        option::PostfixText{ " " },
        option::ForegroundColor{ Color::blue },
        option::FontStyles{ std::vector<FontStyle>{ FontStyle::bold } },
        option::MaxProgress{ 100 },
        option::PrefixText{ preFix },
        option::ShowElapsedTime{ true },
        option::ShowRemainingTime{ true });
    return bar;
  }

  template<typename Type>
  class Wrap
  {
   public:
    Wrap(Type type_value) : m_type_value(type_value)
    {
      // Hide cursor
      indicators::show_console_cursor(false);
    }
    ~Wrap()
    {
      // Show cursor
      indicators::show_console_cursor(true);
    }
    struct Iterator
    {
      using iterator_category = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = typename Type::iterator;
      using pointer = typename Type::iterator::pointer;      // or also value_type*
      using reference = typename Type::iterator::reference;  // or also value_type&
      Iterator(value_type iter, size_t size = 0) : iter_item(iter)
      {
        m_bar->set_option(option::MaxProgress(size));
      }

      reference operator*() const
      {
        return *iter_item;
      }
      value_type operator->()
      {
        return iter_item;
      }

      // Prefix increment
      Iterator operator++()
      {
        m_bar->tick();
        ++iter_item;
        return *this;
      }

      // Postfix increment
      Iterator operator++(int)
      {
        // TODO : How to do this???
        // At this point it is exactly equal to PreFix
        m_bar->tick();
        iter_item++;
        return *this;
      }

      friend bool operator==(const Iterator& a, const Iterator& b)
      {
        return a.iter_item == b.iter_item;
      };
      friend bool operator!=(const Iterator& a, const Iterator& b)
      {
        return a.iter_item != b.iter_item;
      };

     private:
      value_type iter_item;
      std::shared_ptr<ProgressBar> m_bar{ build_bar() };
    };

    Iterator begin()
    {
      return Iterator(m_type_value.begin(), m_type_value.size());
    }

    Iterator end()
    {
      return Iterator(m_type_value.end(), m_type_value.size());
    }

   private:
    Type m_type_value;
  };

  Wrap<std::vector<size_t>> intToRange(size_t size)
  {
    auto vec = std::vector<size_t>(size);
    uint count{ 0 };
    for (auto& item : vec)
    {
      item = count++;
    }
    auto w_vec = vk::Wrap<std::vector<size_t>>(std::move(vec));
    return w_vec;
  }

}  // namespace vk