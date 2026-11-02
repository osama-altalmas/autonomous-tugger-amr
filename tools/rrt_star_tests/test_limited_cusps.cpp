#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tugger_navigation/reeds_shepp_steering.hpp"

using namespace tugger_navigation;

void require(bool condition, const char * message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void inspect(
  const ReedsSheppSteering & steering,
  const Pose2D & start,
  const Pose2D & goal,
  double limit,
  const char * label)
{
  const auto result =
    steering.connectLimited(start, goal, limit);

  require(result.valid, "Truncated connection invalid");

  std::cout << "\n" << label << '\n';
  std::cout << "Limit: " << limit << '\n';
  std::cout << "Length: " << result.total_length << '\n';
  std::cout << "Segments: " << result.segments.size() << '\n';
  std::cout << "Gear changes: " << result.gear_changes << '\n';

  for (std::size_t i = 0; i < result.segments.size(); ++i) {
    const auto & segment = result.segments[i];

    std::cout
      << "  Segment " << i
      << " direction=" << segment.direction
      << " length=" << segment.length
      << '\n';
  }

  double previous_distance = 0.0;
  double max_spacing = 0.0;

  for (const auto & sample : result.samples) {
    require(
      sample.distance >= previous_distance - 1e-8,
      "Non-monotonic sample distances");

    const double spacing =
      sample.distance - previous_distance;

    max_spacing = std::max(max_spacing, spacing);

    previous_distance = sample.distance;
  }

  std::cout << "Max sample spacing: " << max_spacing << '\n';

  require(
    max_spacing <= steering.sampleResolution() + 1e-8,
    "Sample spacing exceeds resolution");

  require(
    std::abs(
      result.samples.back().distance -
      result.total_length) < 1e-8,
    "Final sample distance inconsistent");

  std::cout << "Last sample direction: "
            << result.samples.back().direction << '\n';

  std::cout << "Last sample cusp: "
            << result.samples.back().cusp << '\n';


  int cusp_samples = 0;
  bool found_transition = false;

  for (std::size_t i = 0; i < result.samples.size(); ++i) {
    if (result.samples[i].cusp) {
      ++cusp_samples;
    }

    if (i > 0) {
      const auto & before = result.samples[i - 1];
      const auto & after = result.samples[i];

      if (before.cusp && after.cusp &&
          before.direction != after.direction &&
          std::abs(before.distance - after.distance) < 1e-8) {
        found_transition = true;
      }
    }
  }

  std::cout << "Cusp-marked samples: "
            << cusp_samples << '\n';

  if (result.gear_changes > 0) {
    require(
      cusp_samples >= 2,
      "Direction change lacks explicit cusp samples");

    require(
      found_transition,
      "No paired cusp samples at direction transition");
  }

  std::cout << "Diagnostic checks: PASS\n";
}

int main()
{
  ReedsSheppSteering steering(0.29, 0.05);

  const Pose2D start{0.0, 0.0, 0.0};
  const Pose2D goal{0.0, 0.0, 3.14159265358979323846};

  const auto full = steering.connect(start, goal);

  require(full.valid, "Full connection invalid");
  require(full.gear_changes > 0, "Expected a cusp");

  double cusp_distance = 0.0;
  double accumulated = 0.0;
  int previous_direction = 0;

  for (const auto & segment : full.segments) {
    if (previous_direction != 0 &&
        previous_direction != segment.direction) {
      cusp_distance = accumulated;
      break;
    }

    accumulated += segment.length;
    previous_direction = segment.direction;
  }

  require(cusp_distance > 0.0, "Could not locate cusp");

  std::cout << "Full path length: "
            << full.total_length << '\n';

  std::cout << "First cusp at: "
            << cusp_distance << '\n';

  inspect(
    steering, start, goal,
    cusp_distance - 0.01,
    "BEFORE CUSP");

  inspect(
    steering, start, goal,
    cusp_distance,
    "EXACTLY AT CUSP");

  inspect(
    steering, start, goal,
    cusp_distance + 0.01,
    "AFTER CUSP");

  std::cout << "\nCUSP DIAGNOSTICS COMPLETED\n";

  return 0;
}
