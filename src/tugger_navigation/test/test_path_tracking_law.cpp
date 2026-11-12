#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tugger_navigation/path_tracking_law.hpp"

using tugger_navigation::path_tracking::Point2D;
using tugger_navigation::path_tracking::TrackingGains;
using tugger_navigation::path_tracking::first_section_end;
using tugger_navigation::path_tracking::remembered_signal;
using tugger_navigation::path_tracking::scheduled_lookahead;
using tugger_navigation::path_tracking::track_path;

namespace
{

std::vector<Point2D> straight(double length, double step = 0.01)
{
  std::vector<Point2D> path;
  for (double s = 0.0; s <= length + 1e-9; s += step) {
    path.push_back({s, 0.0});
  }
  return path;
}

// Straight along +x, then a left arc of the given radius.
std::vector<Point2D> straight_then_arc(double radius, double step = 0.01)
{
  auto path = straight(2.0, step);
  for (double a = step / radius; a <= M_PI / 2; a += step / radius) {
    path.push_back({2.0 + radius * std::sin(a), radius - radius * std::cos(a)});
  }
  return path;
}

// Unicycle model at the robot centre driven by the tracking law.
double max_error_on_corner(
  const TrackingGains & gains, double radius, double speed, double start_offset = 0.0,
  double * final_error = nullptr)
{
  const auto path = straight_then_arc(radius);
  double x = 0.0, y = start_offset, yaw = 0.0, worst = 0.0, last = 0.0;
  const double dt = 0.01;
  for (int i = 0; i < 2000; ++i) {
    const auto r = track_path(path, x, y, yaw, speed, gains);
    EXPECT_TRUE(r.valid);
    worst = std::max(worst, std::abs(r.lateral_error));
    last = r.lateral_error;
    x += speed * std::cos(yaw) * dt;
    y += speed * std::sin(yaw) * dt;
    yaw += speed * r.curvature * dt;
    if (std::hypot(x - path.back().x, y - path.back().y) < 0.2) {
      break;
    }
  }
  if (final_error) {
    *final_error = last;
  }
  return worst;
}

}  // namespace

TEST(PathTrackingLaw, OnTheLineGivesZeroCurvature)
{
  const auto r = track_path(straight(3.0), 1.0, 0.0, 0.0, 1.0, TrackingGains{});
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.lateral_error, 0.0, 1e-9);
  EXPECT_NEAR(r.heading_error, 0.0, 1e-9);
  EXPECT_NEAR(r.curvature, 0.0, 1e-9);
}

TEST(PathTrackingLaw, LeftOfThePathTurnsRight)
{
  const auto r = track_path(straight(3.0), 1.0, 0.05, 0.0, 1.0, TrackingGains{});
  EXPECT_NEAR(r.lateral_error, 0.05, 1e-9);
  EXPECT_LT(r.curvature, 0.0);
}

TEST(PathTrackingLaw, HeadingLeftOfThePathTurnsRight)
{
  const auto r = track_path(straight(3.0), 1.0, 0.0, 0.1, 1.0, TrackingGains{});
  EXPECT_NEAR(r.heading_error, 0.1, 1e-9);
  EXPECT_LT(r.curvature, 0.0);
}

TEST(PathTrackingLaw, OnAnArcCommandsTheArcCurvature)
{
  // Robot on a radius 1 m arc, tangent to it: feedforward only.
  std::vector<Point2D> path;
  for (double a = 0.0; a <= M_PI / 2; a += 0.01) {
    path.push_back({std::sin(a), 1.0 - std::cos(a)});
  }
  const double a = 0.5;
  TrackingGains gains;
  gains.preview_time = 0.0;
  const auto r = track_path(path, std::sin(a), 1.0 - std::cos(a), a, 0.6, gains);
  EXPECT_NEAR(r.path_curvature, 1.0, 0.02);
  EXPECT_NEAR(r.curvature, 1.0, 0.03);
}

TEST(PathTrackingLaw, RearFirstTravelUsesTheTravelDirection)
{
  // Path runs along -x; the robot faces +x and drives rear first.
  std::vector<Point2D> path;
  for (double s = 0.0; s <= 3.0; s += 0.01) {
    path.push_back({-s, 0.0});
  }
  // Travel heading is pi; robot at y = +0.05 is right of the travel direction.
  const auto r = track_path(path, -1.0, 0.05, M_PI, -1.0, TrackingGains{});
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.lateral_error, -0.05, 1e-9);
  EXPECT_NEAR(r.heading_error, 0.0, 1e-9);
  // Turn left relative to travel: positive yaw rate.
  EXPECT_GT(r.curvature, 0.0);
}

TEST(PathTrackingLaw, StopsAtACusp)
{
  auto path = straight(1.0);
  for (double s = 0.99; s >= 0.0; s -= 0.01) {
    path.push_back({s, 0.02});
  }
  EXPECT_EQ(first_section_end(path), 101u);
  // Near the cusp the preview must not read the reversal as curvature.
  const auto r = track_path(path, 0.95, 0.0, 0.0, 1.0, TrackingGains{});
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.path_curvature, 0.0, 1e-6);
}

TEST(PathTrackingLaw, FollowsARouteCornerClosely)
{
  // The route corners are 0.7-1.3 m radius, driven at 0.6 m/s. Pure pursuit
  // with the route look-ahead cut 10-17 cm inside them. This model turns
  // without steering lag, so the curvature preview (which covers that lag on
  // the robot) is off here.
  TrackingGains gains;
  gains.preview_time = 0.0;
  EXPECT_LT(max_error_on_corner(gains, 0.74, 0.6), 0.01);
  EXPECT_LT(max_error_on_corner(gains, 1.3, 0.6), 0.01);
}

TEST(PathTrackingLaw, RecoversFromASidewaysOffsetWithoutOvershoot)
{
  // 10 cm off at the start: settles within the 2 m straight, no swing past
  // the path larger than a few millimetres (critically damped gains).
  TrackingGains gains;
  gains.preview_time = 0.0;
  double final_error = 1.0;
  EXPECT_LE(max_error_on_corner(gains, 1.3, 0.6, 0.10, &final_error), 0.1001);
  EXPECT_LT(std::abs(final_error), 0.01);
  const auto path = straight(4.0);
  double x = 0.0, y = 0.10, yaw = 0.0, lowest = 1.0;
  for (int i = 0; i < 600; ++i) {
    const auto r = track_path(path, x, y, yaw, 0.6, gains);
    lowest = std::min(lowest, r.lateral_error);
    x += 0.6 * std::cos(yaw) * 0.01;
    y += 0.6 * std::sin(yaw) * 0.01;
    yaw += 0.6 * r.curvature * 0.01;
  }
  EXPECT_GT(lowest, -0.005);
  EXPECT_LT(std::abs(y), 0.01);
}

TEST(PathTrackingLaw, GentleOnStraightsFirmNearCurves)
{
  const TrackingGains gains;
  // Long straight: straight gains, no planned speed limit.
  const auto far = track_path(straight(5.0), 0.5, 0.0, 0.0, 1.0, gains);
  EXPECT_NEAR(far.lookahead, gains.straight_lookahead, 1e-9);
  EXPECT_TRUE(std::isinf(far.speed_limit));
  // At the start of a radius 0.74 m arc: curve gains.
  const auto path = straight_then_arc(0.74);
  const auto in = track_path(path, 2.0, 0.0, 0.0, 0.6, gains);
  EXPECT_NEAR(in.lookahead, gains.curve_lookahead, 1e-9);
  // Halfway between: blended, never outside the two values.
  const auto mid = track_path(path, 1.3, 0.0, 0.0, 0.6, gains);
  EXPECT_GT(mid.lookahead, gains.curve_lookahead);
  EXPECT_LT(mid.lookahead, gains.straight_lookahead);
  // A correction for the same offset is firmer near the curve.
  const auto off_far = track_path(straight(5.0), 0.5, 0.02, 0.0, 1.0, gains);
  const auto off_in = track_path(path, 2.0, 0.02, 0.0, 0.6, gains);
  EXPECT_LT(off_in.curvature - off_in.path_curvature, off_far.curvature);
}

TEST(PathTrackingLaw, ScheduledLookaheadIsMonotonic)
{
  const TrackingGains gains;
  double previous = scheduled_lookahead(0.0, gains);
  EXPECT_NEAR(previous, gains.straight_lookahead, 1e-12);
  for (double k = 0.05; k <= 1.0; k += 0.05) {
    const double l = scheduled_lookahead(k, gains);
    EXPECT_LE(l, previous + 1e-12);
    previous = l;
  }
  EXPECT_NEAR(previous, gains.curve_lookahead, 1e-12);
}

TEST(PathTrackingLaw, PassedCurveFadesOverTheConfiguredDistance)
{
  const TrackingGains gains;
  EXPECT_NEAR(remembered_signal(1.0, 0.0, gains), 1.0, 1e-12);
  EXPECT_NEAR(remembered_signal(1.0, gains.schedule_behind / 2, gains), 0.5, 1e-12);
  EXPECT_NEAR(remembered_signal(1.0, gains.schedule_behind, gains), 0.0, 1e-12);
  EXPECT_NEAR(remembered_signal(1.0, 5.0, gains), 0.0, 1e-12);
}

TEST(PathTrackingLaw, PlansTheCurveSpeedBeforeTheCurve)
{
  const TrackingGains gains;
  const auto path = straight_then_arc(0.74);
  // Inside the margin before the curve and in it: the curve speed.
  EXPECT_NEAR(track_path(path, 1.9, 0.0, 0.0, 0.6, gains).speed_limit, gains.curve_speed, 0.01);
  EXPECT_NEAR(track_path(path, 2.0, 0.0, 0.0, 0.6, gains).speed_limit, gains.curve_speed, 0.01);
  // Further back: the braking profile v^2 = v_curve^2 + 2 a d.
  const auto before = track_path(path, 1.0, 0.0, 0.0, 1.0, gains);
  const double d = (2.0 - 1.0) - gains.speed_margin;
  EXPECT_NEAR(
    before.speed_limit,
    std::sqrt(gains.curve_speed * gains.curve_speed + 2.0 * gains.speed_deceleration * d), 0.06);
  // Far from any curve: no limit.
  EXPECT_TRUE(std::isinf(track_path(straight(6.0), 0.5, 0.0, 0.0, 1.0, gains).speed_limit));
}

TEST(PathTrackingLaw, DegeneratePathIsInvalid)
{
  EXPECT_FALSE(track_path({}, 0.0, 0.0, 0.0, 1.0, TrackingGains{}).valid);
  EXPECT_FALSE(track_path({{1.0, 1.0}}, 0.0, 0.0, 0.0, 1.0, TrackingGains{}).valid);
  EXPECT_FALSE(
    track_path({{1.0, 1.0}, {1.0, 1.0}}, 0.0, 0.0, 0.0, 1.0, TrackingGains{}).valid);
}
