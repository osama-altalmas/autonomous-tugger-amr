#ifndef TUGGER_NAVIGATION__PLANNING_INTERRUPTED_HPP_
#define TUGGER_NAVIGATION__PLANNING_INTERRUPTED_HPP_

#include <stdexcept>

namespace tugger_navigation
{

class PlanningInterrupted : public std::runtime_error
{
public:
  PlanningInterrupted()
  : std::runtime_error("RRT* planning interrupted")
  {
  }
};

}  // namespace tugger_navigation

#endif
