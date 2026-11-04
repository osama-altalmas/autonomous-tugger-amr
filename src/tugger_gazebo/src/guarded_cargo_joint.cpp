// Command-driven, initially detached Gazebo cargo joint for the tugger.
// This is simulator-only; it does not certify cargo clearance or lift contact.
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <gz/msgs/empty.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace tugger_gazebo
{
class GuardedCargoJoint : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity & entity,
                 const std::shared_ptr<const sdf::Element> & sdf,
                 gz::sim::EntityComponentManager & ecm,
                 gz::sim::EventManager &) override
  {
    gz::sim::Model model(entity);
    if (!model.Valid(ecm) || !sdf->HasElement("parent_link") ||
        !sdf->HasElement("child_model") || !sdf->HasElement("child_link") ||
        !sdf->HasElement("attach_topic") || !sdf->HasElement("detach_topic") ||
        !sdf->HasElement("output_topic")) {
      return;
    }
    parent_ = model.LinkByName(ecm, sdf->Get<std::string>("parent_link"));
    child_model_ = sdf->Get<std::string>("child_model");
    child_link_ = sdf->Get<std::string>("child_link");
    if (parent_ == gz::sim::kNullEntity) {
      return;
    }
    const auto attach_topic = sdf->Get<std::string>("attach_topic");
    const auto detach_topic = sdf->Get<std::string>("detach_topic");
    const auto output_topic = sdf->Get<std::string>("output_topic");
    state_pub_ = node_.Advertise<gz::msgs::StringMsg>(output_topic);
    if (!state_pub_ || !node_.Subscribe(attach_topic, &GuardedCargoJoint::OnAttach, this) ||
        !node_.Subscribe(detach_topic, &GuardedCargoJoint::OnDetach, this)) {
      return;
    }
    valid_ = true;
  }

  void PreUpdate(const gz::sim::UpdateInfo & info,
                 gz::sim::EntityComponentManager & ecm) override
  {
    if (!valid_ || info.paused) {
      return;
    }
    const auto parent_pose = gz::sim::worldPose(parent_, ecm);
    const double now = std::chrono::duration<double>(info.simTime).count();
    if (now < last_state_time_ || now - last_state_time_ >= 1.0) {
      Publish(attached_ ? "attached" : "detached");
      last_state_time_ = now;
    }
    if (!have_anchor_ || now < anchor_time_ ||
        parent_pose.Pos().Distance(anchor_pose_.Pos()) > 0.001 ||
        std::abs(std::remainder(parent_pose.Rot().Yaw() - anchor_pose_.Rot().Yaw(),
                                2.0 * M_PI)) > 0.005) {
      anchor_pose_ = parent_pose;
      anchor_time_ = now;
      have_anchor_ = true;
    }
    if (detach_requested_.exchange(false)) {
      if (attached_ && now - anchor_time_ >= 0.5) {
        ecm.RequestRemoveEntity(joint_);
        joint_ = gz::sim::kNullEntity;
        attached_ = false;
        Publish("detached");
      } else {
        Publish("detach_rejected");
      }
    }
    if (!attach_requested_.exchange(false)) {
      return;
    }
    if (attached_ || now - anchor_time_ < 0.5) {
      Publish("attach_rejected");
      return;
    }
    const auto matches = gz::sim::entitiesFromScopedName(child_model_, ecm);
    if (matches.size() != 1) {
      Publish("attach_rejected");
      return;
    }
    gz::sim::Model child(*matches.begin());
    const auto child_entity = child.LinkByName(ecm, child_link_);
    if (!child.Valid(ecm) || child_entity == gz::sim::kNullEntity) {
      Publish("attach_rejected");
      return;
    }
    const auto child_pose = gz::sim::worldPose(child_entity, ecm);
    const auto delta = child_pose.Pos() - parent_pose.Pos();
    const double planar = std::hypot(delta.X(), delta.Y());
    // Current simulated cargo has a 0.28 m underside and the lift platform
    // is 15 mm thick. Reject a joint request outside the physical capture band.
    const double vertical_gap = child_pose.Pos().Z() + 0.28 -
                                (parent_pose.Pos().Z() + 0.0075);
    const double axis_error = std::abs(std::remainder(
      child_pose.Rot().Yaw() - parent_pose.Rot().Yaw(), M_PI / 2.0));
    if (planar > 0.03 || vertical_gap < -0.01 || vertical_gap > 0.06 ||
        axis_error > 0.05) {
      Publish("attach_rejected");
      return;
    }
    joint_ = ecm.CreateEntity();
    ecm.CreateComponent(joint_, gz::sim::components::DetachableJoint(
      {parent_, child_entity, "fixed"}));
    attached_ = true;
    Publish("attached");
  }

private:
  void OnAttach(const gz::msgs::Empty &) { attach_requested_ = true; }
  void OnDetach(const gz::msgs::Empty &) { detach_requested_ = true; }
  void Publish(const std::string & text)
  {
    gz::msgs::StringMsg message;
    message.set_data(text);
    state_pub_.Publish(message);
  }
  gz::transport::Node node_;
  gz::transport::Node::Publisher state_pub_;
  gz::sim::Entity parent_{gz::sim::kNullEntity};
  gz::sim::Entity joint_{gz::sim::kNullEntity};
  std::string child_model_;
  std::string child_link_;
  std::atomic<bool> attach_requested_{false};
  std::atomic<bool> detach_requested_{false};
  bool valid_{false};
  bool attached_{false};
  bool have_anchor_{false};
  double anchor_time_{0.0};
  double last_state_time_{-1.0};
  gz::math::Pose3d anchor_pose_;
};
}  // namespace tugger_gazebo

GZ_ADD_PLUGIN(tugger_gazebo::GuardedCargoJoint, gz::sim::System,
              tugger_gazebo::GuardedCargoJoint::ISystemConfigure,
              tugger_gazebo::GuardedCargoJoint::ISystemPreUpdate)
