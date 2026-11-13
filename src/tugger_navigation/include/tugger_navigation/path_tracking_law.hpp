// Kinematic path tracking for the 4WS tugger: path-curvature feedforward plus
// lateral and heading feedback, evaluated at the robot centre.
//
// With counter-phase steering the robot centre moves along its own heading (no
// side slip), so the unicycle path-tracking model applies there:
//   kappa_cmd = kappa * cos(eh) / (1 - kappa * e) - k_e * e * sinc(eh) - k_h * eh
// where e is the sideways offset (left of the path, travel direction), eh the
// heading error and kappa the path curvature. Linearised, e'' + k_h e' + k_e e = 0
// per metre travelled, so the error settles with the same distance constant at
// every speed.
//
// Pure pursuit lacks the feedforward term: it aims at a point one look-ahead
// ahead, so on the 1-1.5 m long route corners it starts turning early, never
// reaches the path curvature and cuts 10-17 cm inside (measured 2026-11-12).
//
// The correction strength follows the path ahead: gentle on straights (like a
// 1.2 m look-ahead, so small localization jumps barely move the steering),
// firm approaching and inside curves (like a 0.6 m look-ahead). For an
// equivalent look-ahead L the gains are k_e = 2 / L^2 and k_h = 2 sqrt(2) / L
// (critically damped). Shorter than about 0.6 m amplified localization noise
// into steering reversals in the offline model (0.3 m: 8 per 10 m).
//
// The curve speed is planned from the path: the robot reaches the curve speed
// a margin before the first point that needs it, instead of braking inside
// the curve once the command already turns.
//
// Pure functions only (no ROS types) so they can be unit tested.

#ifndef TUGGER_NAVIGATION__PATH_TRACKING_LAW_HPP_
#define TUGGER_NAVIGATION__PATH_TRACKING_LAW_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tugger_navigation
{
namespace path_tracking
{

struct Point2D
{
  double x;
  double y;
};

struct TrackingGains
{
  double straight_lookahead = 1.2;  // m, equivalent look-ahead on straights
  double curve_lookahead = 0.6;     // m, equivalent look-ahead in curves
  double schedule_low_curvature = 0.1;   // 1/m, at or below: straight gains
  double schedule_high_curvature = 0.5;  // 1/m, at or above: curve gains
  double schedule_ahead = 1.0;      // m of path ahead that sets the gains
  double schedule_behind = 0.5;     // m travelled before a passed curve stops counting
  double preview_time = 0.10;  // s, covers the steering slew and control delay
  double curvature_window = 0.05;  // m, half window for the path curvature
  double max_curvature = 1.96;  // 1/m, Ackermann at 45 deg on the inner wheel (0.51 m)
  double curve_speed = 0.6;    // m/s, curve speed (motion gate curve limit)
  double curve_speed_curvature = 0.25;  // 1/m, path curvature that needs it
  double speed_deceleration = 1.0;  // m/s^2, planned braking toward a curve
  double speed_margin = 0.2;   // m, reach the curve speed this far before it
  double speed_scan = 2.0;     // m of path ahead searched for the next curve
};

struct TrackingResult
{
  bool valid = false;
  double lateral_error = 0.0;   // m, robot left of the path is positive
  double heading_error = 0.0;   // rad, travel heading minus path tangent
  double path_curvature = 0.0;  // 1/m, at the preview point
  double curvature = 0.0;       // 1/m, commanded, travel frame
  double schedule_signal = 0.0;  // 1/m, weighted largest curvature near the robot
  double lookahead = 0.0;       // m, equivalent look-ahead used for the gains
  double speed_limit = 0.0;     // m/s, planned limit (infinity when none)
  double path_ahead = 0.0;      // m of path available ahead of the robot
};

inline double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

// Index one past the last point of the first travel section: the path ends
// there or reverses direction (a cusp) after it.
inline std::size_t first_section_end(const std::vector<Point2D> & path)
{
  double px = 0.0;
  double py = 0.0;
  bool have_previous = false;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const double dx = path[i].x - path[i - 1].x;
    const double dy = path[i].y - path[i - 1].y;
    if (std::hypot(dx, dy) < 1e-6) {
      continue;
    }
    if (have_previous && px * dx + py * dy < 0.0) {
      return i;
    }
    px = dx;
    py = dy;
    have_previous = true;
  }
  return path.size();
}

// Index of the non-degenerate segment (path[j - 1] -> path[j]) containing arc
// length s, clamped to the section.
inline std::size_t segment_at(
  const std::vector<double> & arc, std::size_t end, double s)
{
  std::size_t i = 1;
  while (i + 1 < end && arc[i] < s) {
    ++i;
  }
  // Skip zero-length segments backwards, then forwards.
  std::size_t j = i;
  while (j > 1 && arc[j] - arc[j - 1] < 1e-9) {
    --j;
  }
  if (arc[j] - arc[j - 1] < 1e-9) {
    j = i;
    while (j + 1 < end && arc[j] - arc[j - 1] < 1e-9) {
      ++j;
    }
  }
  return j;
}

inline double segment_tangent(const std::vector<Point2D> & path, std::size_t j)
{
  return std::atan2(path[j].y - path[j - 1].y, path[j].x - path[j - 1].x);
}

// Path curvature at arc length s: tangent change between the segments at
// s - window and s + window, divided by the distance between their midpoints.
inline double curvature_at(
  const std::vector<Point2D> & path, const std::vector<double> & arc,
  std::size_t end, double s, double window)
{
  const std::size_t ja = segment_at(arc, end, std::max(0.0, s - window));
  const std::size_t jb = segment_at(arc, end, std::min(arc[end - 1], s + window));
  const double distance = 0.5 * (arc[jb] + arc[jb - 1] - arc[ja] - arc[ja - 1]);
  if (jb == ja || distance < 1e-3) {
    return 0.0;
  }
  return wrap_angle(segment_tangent(path, jb) - segment_tangent(path, ja)) / distance;
}

// Equivalent look-ahead for a schedule signal (largest weighted curvature).
inline double scheduled_lookahead(double signal, const TrackingGains & gains)
{
  const double span = gains.schedule_high_curvature - gains.schedule_low_curvature;
  const double f = span > 1e-9 ?
    std::clamp((signal - gains.schedule_low_curvature) / span, 0.0, 1.0) :
    (signal >= gains.schedule_high_curvature ? 1.0 : 0.0);
  return gains.straight_lookahead + (gains.curve_lookahead - gains.straight_lookahead) * f;
}

// A passed curve keeps counting while the robot travels schedule_behind
// metres, fading linearly, so the gains relax smoothly after a curve even
// though the plan behind the robot is pruned.
inline double remembered_signal(
  double previous_signal, double travelled, const TrackingGains & gains)
{
  if (gains.schedule_behind <= 0.0) {
    return 0.0;
  }
  return previous_signal * std::max(0.0, 1.0 - travelled / gains.schedule_behind);
}

// path: plan positions in travel order, in the same frame as the robot pose.
// travel_yaw: robot heading, plus pi when the robot drives rear first.
// previous_signal: schedule signal of the last cycle after remembered_signal()
// (0 for a new path).
inline TrackingResult track_path(
  const std::vector<Point2D> & path, double robot_x, double robot_y,
  double travel_yaw, double speed, const TrackingGains & gains,
  double previous_signal = 0.0)
{
  TrackingResult result;
  const std::size_t end = first_section_end(path);
  if (end < 2) {
    return result;
  }
  std::vector<double> arc(end, 0.0);
  for (std::size_t i = 1; i < end; ++i) {
    arc[i] = arc[i - 1] + std::hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
  }
  if (arc[end - 1] < 1e-3) {
    return result;
  }

  // Closest point on the section.
  double best = 1e18;
  double best_s = 0.0;
  std::size_t best_seg = 1;
  for (std::size_t i = 1; i < end; ++i) {
    const double sx = path[i].x - path[i - 1].x;
    const double sy = path[i].y - path[i - 1].y;
    const double len = arc[i] - arc[i - 1];
    if (len < 1e-9) {
      continue;
    }
    const double t = std::clamp(
      ((robot_x - path[i - 1].x) * sx + (robot_y - path[i - 1].y) * sy) / (len * len),
      0.0, 1.0);
    const double dx = robot_x - (path[i - 1].x + t * sx);
    const double dy = robot_y - (path[i - 1].y + t * sy);
    const double d2 = dx * dx + dy * dy;
    if (d2 < best) {
      best = d2;
      best_s = arc[i - 1] + t * len;
      best_seg = i;
    }
  }

  const double tangent = segment_tangent(path, best_seg);
  const Point2D & a = path[best_seg - 1];
  // Signed distance to the tangent line through the closest segment.
  const double lateral = -std::sin(tangent) * (robot_x - a.x) + std::cos(tangent) * (robot_y - a.y);
  const double heading = wrap_angle(travel_yaw - tangent);
  const double preview = best_s + std::abs(speed) * gains.preview_time;
  const double kappa = curvature_at(path, arc, end, preview, gains.curvature_window);

  // Path ahead, sampled every 5 cm: gain schedule and planned curve speed.
  const double step = 0.05;
  const double horizon = std::max(gains.schedule_ahead, gains.speed_scan);
  double signal = previous_signal;
  double first_curve = -1.0;
  for (double d = 0.0; d <= horizon + 1e-9 && best_s + d <= arc[end - 1] + 1e-9; d += step) {
    const double k = std::abs(curvature_at(path, arc, end, best_s + d, gains.curvature_window));
    if (d <= gains.schedule_ahead && gains.schedule_ahead > 0.0) {
      signal = std::max(signal, k * (1.0 - d / gains.schedule_ahead));
    }
    if (first_curve < 0.0 && d <= gains.speed_scan && k >= gains.curve_speed_curvature) {
      first_curve = d;
    }
  }
  const double lookahead = std::max(scheduled_lookahead(signal, gains), 1e-3);
  const double lateral_gain = 2.0 / (lookahead * lookahead);
  const double heading_gain = 2.0 * std::sqrt(2.0) / lookahead;

  const double sinc = std::abs(heading) > 1e-6 ? std::sin(heading) / heading : 1.0;
  double command = kappa * std::cos(heading) / std::max(1.0 - kappa * lateral, 0.2) -
    lateral_gain * lateral * sinc - heading_gain * heading;
  command = std::clamp(command, -gains.max_curvature, gains.max_curvature);

  double speed_limit = std::numeric_limits<double>::infinity();
  if (first_curve >= 0.0) {
    const double distance = std::max(0.0, first_curve - gains.speed_margin);
    speed_limit = std::sqrt(
      gains.curve_speed * gains.curve_speed + 2.0 * gains.speed_deceleration * distance);
  }

  result.valid = true;
  result.lateral_error = lateral;
  result.heading_error = heading;
  result.path_curvature = kappa;
  result.curvature = command;
  result.schedule_signal = signal;
  result.lookahead = lookahead;
  result.speed_limit = speed_limit;
  result.path_ahead = arc[end - 1] - best_s;
  return result;
}

}  // namespace path_tracking
}  // namespace tugger_navigation

#endif  // TUGGER_NAVIGATION__PATH_TRACKING_LAW_HPP_
