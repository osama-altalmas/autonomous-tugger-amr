// Simulator-only collision-scene observation. Never publishes motion commands.
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <gz/math/Matrix3.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/transport/Node.hh>
#include <nlohmann/json.hpp>
#include <sdf/Box.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Sphere.hh>

namespace tugger_gazebo
{
class CollisionScene : public gz::sim::System,
                       public gz::sim::ISystemConfigure,
                       public gz::sim::ISystemPostUpdate
{
public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override
  {
    publisher_ = node_.Advertise<gz::msgs::StringMsg>("/tugger/collision_scene");
  }

  void PostUpdate(const gz::sim::UpdateInfo & info,
                  const gz::sim::EntityComponentManager & ecm) override
  {
    if (info.paused || !publisher_) return;
    const double now = std::chrono::duration<double>(info.simTime).count();
    if (now >= last_ && now - last_ < 0.1) return;
    last_ = now;
    nlohmann::json scene = {{"stamp", now}, {"valid", true},
                            {"boxes", nlohmann::json::array()}};
    ecm.Each<gz::sim::components::Collision>(
      [&](const gz::sim::Entity & entity, const gz::sim::components::Collision *) {
        std::string model;
        auto current = entity;
        while (current != gz::sim::kNullEntity) {
          if (ecm.Component<gz::sim::components::Model>(current)) {
            auto name = ecm.Component<gz::sim::components::Name>(current);
            if (name) model = name->Data();
            break;
          }
          auto parent = ecm.Component<gz::sim::components::ParentEntity>(current);
          if (!parent) break;
          current = parent->Data();
        }
        if (model == "autonomous_tugger") return true;
        const auto geometry = ecm.Component<gz::sim::components::Geometry>(entity);
        if (!geometry || model.empty()) {
          scene["valid"] = false;
          return true;
        }
        const auto & shape = geometry->Data();
        if (shape.Type() == sdf::GeometryType::PLANE && model == "ground_plane") return true;
        gz::math::Vector3d half;
        if (shape.Type() == sdf::GeometryType::BOX && shape.BoxShape()) {
          half = shape.BoxShape()->Size() * 0.5;
        } else if (shape.Type() == sdf::GeometryType::SPHERE && shape.SphereShape()) {
          const double r = shape.SphereShape()->Radius();
          half.Set(r, r, r);
        } else if (shape.Type() == sdf::GeometryType::CYLINDER && shape.CylinderShape()) {
          const double r = shape.CylinderShape()->Radius();
          half.Set(r, r, shape.CylinderShape()->Length() * 0.5);
        } else {
          scene["valid"] = false;  // Unsupported collision: do not silently ignore it.
          return true;
        }
        const auto pose = gz::sim::worldPose(entity, ecm);
        const gz::math::Matrix3d rotation(pose.Rot());
        double extents[3] = {};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) extents[i] += std::abs(rotation(i, j)) * half[j];
        }
        if (pose.Pos().Z() + extents[2] <= 1e-6) return true;
        scene["boxes"].push_back({{"model", model},
          {"name", gz::sim::scopedName(entity, ecm, "::", false)},
          {"x", pose.Pos().X()}, {"y", pose.Pos().Y()}, {"z", pose.Pos().Z()},
          {"hx", extents[0]}, {"hy", extents[1]}, {"hz", extents[2]}});
        return true;
      });
    gz::msgs::StringMsg msg;
    msg.set_data(scene.dump());
    publisher_.Publish(msg);
  }
private:
  gz::transport::Node node_;
  gz::transport::Node::Publisher publisher_;
  double last_{-1.0};
};
}  // namespace tugger_gazebo

GZ_ADD_PLUGIN(tugger_gazebo::CollisionScene, gz::sim::System,
              tugger_gazebo::CollisionScene::ISystemConfigure,
              tugger_gazebo::CollisionScene::ISystemPostUpdate)
