#!/usr/bin/env bash
# Download the Gazebo Fuel models used for the obstacle dataset into Gazebo's
# own cache (~/.gz/fuel, about 200 MB; not in the repository). Licences: CC0 or
# CC-BY-4.0 (credits in README.md, section "Obstacle recognition (AI)").
set -euo pipefail
set +u; source /opt/ros/lyrical/setup.bash; set -u
B=https://fuel.gazebosim.org/1.0
for m in \
  "OpenRobotics/models/Standing person" "OpenRobotics/models/Walking person" \
  "OpenRobotics/models/Scrubs" "OpenRobotics/models/OpScrubs" "OpenRobotics/models/Casual female" \
  "OpenRobotics/models/Euro pallet" "jfrascon/models/Euro_pallet_w_stacked_boxes" \
  "OpenRobotics/models/Pallet_Standard" \
  "OpenRobotics/models/Utility Cart" "OpenRobotics/models/Hand_Cart" "OpenRobotics/models/BMWCart" \
  "OpenRobotics/models/TrolleyBox1" "OpenRobotics/models/BPCart" "OpenRobotics/models/ParkingTrolleyMin" \
  "OpenRobotics/models/Forklift" "OpenRobotics/models/Pallet_Jack" \
  "OpenRobotics/models/Cart: rigid suspension" "OpenRobotics/models/MiRCart" \
  "OpenRobotics/models/TrolleyBox2" "OpenRobotics/models/SurgicalTrolley" "OpenRobotics/models/MopCart1" \
  "OpenRobotics/models/Cardboard box" "OpenRobotics/models/Construction Cone" \
  "OpenRobotics/models/Construction Barrel" "OpenRobotics/models/TrashBin" \
  "OpenRobotics/models/Large Crate" \
  "OpenRobotics/models/MaleVisitorOnPhone" "OpenRobotics/models/FemaleVisitor" \
  "OpenRobotics/models/Nurse" "OpenRobotics/models/Survivor Male" "OpenRobotics/models/Survivor Female" \
  "OpenRobotics/models/Handtruck_01" "OpenRobotics/models/Handtruck_02"; do
  gz fuel download -u "$B/${m// /%20}" > /dev/null && echo "ok  $m"
done
