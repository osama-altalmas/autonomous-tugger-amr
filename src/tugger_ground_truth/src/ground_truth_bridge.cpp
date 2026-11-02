#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>

class GroundTruthBridge : public rclcpp::Node
{
public:
  GroundTruthBridge()
  : Node("tugger_ground_truth_bridge")
  {
    robot_name_ = declare_parameter<std::string>(
      "robot_name", "autonomous_tugger");

    gazebo_topic_ = declare_parameter<std::string>(
      "gazebo_topic",
      "/world/tugger_test/dynamic_pose/info");

    publisher_ =
      create_publisher<geometry_msgs::msg::PoseStamped>(
        "/tugger/ground_truth_pose", 10);

    bool subscribed = gazebo_node_.Subscribe(
      gazebo_topic_,
      &GroundTruthBridge::onGazeboPose,
      this);

    if (!subscribed) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to subscribe to Gazebo topic: %s",
        gazebo_topic_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "Listening for Gazebo entity '%s' on %s",
        robot_name_.c_str(),
        gazebo_topic_.c_str());
    }
  }

private:
  void onGazeboPose(const gz::msgs::Pose_V & message)
  {
    for (const auto & pose : message.pose()) {
      if (pose.name() != robot_name_) {
        continue;
      }

      geometry_msgs::msg::PoseStamped output;

      output.header.stamp = now();
      output.header.frame_id = "map";

      output.pose.position.x = pose.position().x();
      output.pose.position.y = pose.position().y();
      output.pose.position.z = pose.position().z();

      output.pose.orientation.x = pose.orientation().x();
      output.pose.orientation.y = pose.orientation().y();
      output.pose.orientation.z = pose.orientation().z();
      output.pose.orientation.w = pose.orientation().w();

      publisher_->publish(output);
      return;
    }
  }

  std::string robot_name_;
  std::string gazebo_topic_;

  gz::transport::Node gazebo_node_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<GroundTruthBridge>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
