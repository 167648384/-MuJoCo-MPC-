
#include "mjpc/tasks/simple_car/simple_car.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <memory>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string SimpleCar::XmlPath() const {
  return GetModelPath("simple_car/task.xml");
}

std::string SimpleCar::Name() const { return "SimpleCar"; }

void SimpleCar::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  // ---------- Position (x, y) ----------
  residual[0] = data->qpos[0] - data->mocap_pos[0];
  residual[1] = data->qpos[1] - data->mocap_pos[1];

  // ---------- Control ----------
  residual[2] = data->ctrl[0];
  residual[3] = data->ctrl[1];
}

void SimpleCar::TransitionLocked(mjModel* model, mjData* data) {
  double car_pos[2] = {data->qpos[0], data->qpos[1]};
  double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
  
  double car_to_goal[2];
  mju_sub(car_to_goal, goal_pos, car_pos, 2);
  
  if (mju_norm(car_to_goal, 2) < 0.2) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[2] = 0.01;
  }
}

void SimpleCar::ModifyScene(const mjModel* model, const mjData* data,
                             mjvScene* scene) const {
  // 这个函数现在不用于渲染仪表盘
  // 仪表盘渲染在 simulate.cc 的2D覆盖层中处理
}

}  // namespace mjpc

