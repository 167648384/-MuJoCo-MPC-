
#ifndef MJPC_TASKS_SIMPLE_CAR_DASHBOARD_H_
#define MJPC_TASKS_SIMPLE_CAR_DASHBOARD_H_

#include <mujoco/mujoco.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mjpc {

// 仪表盘数据结构
struct DashboardData {
    double speed_kmh;     // 速度 (km/h)
    double rpm;           // 转速 (RPM)
    double fuel;          // 油量 (%)
    double temperature;   // 温度 (°C)
    int gear;             // 当前档位
    double position_x;    // X位置
    double position_y;    // Y位置
    double position_z;    // Z位置
};

// 数据提取器
class DashboardDataExtractor {
public:
    DashboardDataExtractor(const mjModel* model) : m_(model) {
        // 查找小车body的ID
        car_body_id_ = mj_name2id(model, mjOBJ_BODY, "car");
        if (car_body_id_ < 0) {
            car_body_id_ = 0;  // 默认为第一个body
        }
        
        // 查找前进电机ID
        motor_forward_id_ = mj_name2id(model, mjOBJ_ACTUATOR, "forward");
        
        // 初始化变量
        last_speed_ms_ = 0;
        fuel_level_ = 100.0;
        current_gear_ = 1;
        last_rpm_ = 100.0;  // 初始设置为很低的转速
        
        // 档位参数
        gear_ratios_[0] = 0;      // 空挡
        gear_ratios_[1] = 4.0;    // 1档
        gear_ratios_[2] = 2.5;    // 2档
        gear_ratios_[3] = 1.7;    // 3档
        gear_ratios_[4] = 1.3;    // 4档
        gear_ratios_[5] = 1.0;    // 5档
        gear_ratios_[6] = 0.8;    // 6档
        
        // 修改最大速度限制为 10 km/h
        max_speed_kmh_ = 10.0;
        max_rpm_ = 1200.0;  // 修改最大转速为 1200 RPM（更合理的值）
    }

    void Update(const mjData* data, DashboardData& dashboard) {
        // 获取车身速度
        int body_vel_index = 6 * car_body_id_;
        double vx = data->cvel[body_vel_index];
        double vy = data->cvel[body_vel_index + 1];
        double speed_ms = sqrt(vx * vx + vy * vy);
        dashboard.speed_kmh = speed_ms * 3.6;
        
        // 限制速度范围（使用新的最大速度）
        if (dashboard.speed_kmh > max_speed_kmh_) dashboard.speed_kmh = max_speed_kmh_;
        if (dashboard.speed_kmh < 0) dashboard.speed_kmh = 0;
        
        // 获取油门输入
        double throttle = 0.0;
        if (motor_forward_id_ >= 0) {
            double ctrl = data->ctrl[motor_forward_id_];
            throttle = fabs(ctrl);  // 取绝对值，防止负值
        }
        
        // ---------- 修正的转速计算逻辑 ----------
        const double idle_rpm = 600.0;      // 降低怠速到600 RPM
        
        // 1. 计算加速度（用于换挡逻辑）
        double acceleration = 0.0;
        if (m_->opt.timestep > 0) {
            acceleration = (speed_ms - last_speed_ms_) / m_->opt.timestep;
        }
        last_speed_ms_ = speed_ms;
        
        // 2. 先使用上一次的转速进行换挡决策
        UpdateGear(dashboard.speed_kmh, throttle, acceleration, last_rpm_);
        
        // 3. 基础转速计算
        double gear_multiplier = gear_ratios_[1] / gear_ratios_[current_gear_];
        
        // 发动机基础转速 = 速度 × 传动比 × 档位修正
        double engine_rpm = 0.0;
        if (current_gear_ > 0) {  // 不在空挡
            // 最小车速对应转速（防止零速度时转速为零）
            double min_speed_rpm = 50.0;  // 降低到50 RPM
            engine_rpm = fmax(dashboard.speed_kmh, min_speed_rpm) * base_gear_ratio * gear_multiplier;
        }
        
        // 4. 添加油门影响
        // 空挡时：只有油门影响转速
        // 行驶时：油门增加额外转速
        double throttle_rpm = 0.0;
        double total_rpm = 0.0;
        
        if (current_gear_ == 0) {
            // 空挡：转速由怠速 + 油门决定
            // 如果没有油门输入，使用很低的怠速
            if (throttle < 0.05) {  // 油门很小
                throttle_rpm = 0;
                engine_rpm = 0;
                total_rpm = 100.0;  // 很低的空挡转速
            } else {
                throttle_rpm = throttle * 500.0;  // 空挡最大增加500 RPM
                total_rpm = idle_rpm + throttle_rpm;
            }
        } else {
            // 行驶：油门增加额外转速
            throttle_rpm = throttle * 400.0;  // 行驶最大增加400 RPM
            total_rpm = idle_rpm + engine_rpm + throttle_rpm;
        }
        
        // 5. 加速度影响（轻微调整）
        if (acceleration > 0.1) {
            // 加速时略微增加转速
            total_rpm *= 1.0 + acceleration * 0.05;
        } else if (acceleration < -0.2) {
            // 减速时略微降低转速
            total_rpm *= 1.0 + acceleration * 0.02;
        }
        
        // 6. 限制转速范围
        if (total_rpm < 0) total_rpm = 0;
        if (total_rpm > max_rpm_) total_rpm = max_rpm_;
        
        // 7. 平滑转速变化（防止突变）
        double rpm_change = total_rpm - last_rpm_;
        if (fabs(rpm_change) > 150) {  // 减小最大变化率
            total_rpm = last_rpm_ + copysign(150, rpm_change);
        }
        
        dashboard.rpm = total_rpm;
        last_rpm_ = total_rpm;  // 保存当前转速供下一次使用
        
        // ---------- 油量计算 ----------
        // 油耗 = 基础油耗 + 转速相关油耗 + 速度相关油耗
        double fuel_consumption = 0.001 
                                + (dashboard.rpm / max_rpm_) * 0.002 
                                + (dashboard.speed_kmh / max_speed_kmh_) * 0.001;
        
        fuel_level_ -= fuel_consumption;
        if (fuel_level_ < 0) fuel_level_ = 0;
        dashboard.fuel = fuel_level_;
        
        // ---------- 温度计算 ----------
        // 温度 = 基础温度 + 转速影响 + 速度影响
        double base_temp = 60.0;
        double rpm_temp = (dashboard.rpm / max_rpm_) * 40.0;
        double speed_temp = (dashboard.speed_kmh / max_speed_kmh_) * 10.0;
        
        dashboard.temperature = base_temp + rpm_temp + speed_temp;
        
        // 温度限制
        if (dashboard.temperature > 120.0) dashboard.temperature = 120.0;
        if (dashboard.temperature < 60.0) dashboard.temperature = 60.0;
        
        // 获取小车位置
        dashboard.position_x = data->xpos[3 * car_body_id_];
        dashboard.position_y = data->xpos[3 * car_body_id_ + 1];
        dashboard.position_z = data->xpos[3 * car_body_id_ + 2];
    }

private:
    void UpdateGear(double speed_kmh, double throttle, double acceleration, double current_rpm) {
        // 简单的自动换挡逻辑
        if (current_gear_ == 0) {
            // 空挡：根据油门决定是否挂1档
            if (throttle > 0.3 && speed_kmh < 2) {
                current_gear_ = 1;
            }
            return;
        }
        
        // 调整换挡速度阈值以适应10 km/h的最大速度
        double speed_thresholds[] = {0, 1.5, 3.0, 4.5, 6.0, 8.0, 9.5};  // 各档位建议升档速度
        double min_speed_thresholds[] = {0, 0.5, 2.0, 3.5, 5.0, 7.0, 8.5};  // 各档位最低速度
        
        // 升档条件：高转速且不处于急加速状态
        if (current_gear_ < 6) {
            bool should_upshift = false;
            
            // 基于速度的升档
            if (speed_kmh > speed_thresholds[current_gear_]) {
                should_upshift = true;
            }
            
            // 高转速保护升档（调整为1200 max）
            if (current_rpm > 1100) {  // 从1300调整为1100
                should_upshift = true;
            }
            
            if (should_upshift && acceleration < 2.0) {  // 不处于急加速状态
                current_gear_++;
                // 升档时转速下降
                last_rpm_ *= 0.7;
            }
        }
        
        // 降档条件：低转速或需要急加速
        if (current_gear_ > 1) {
            bool should_downshift = false;
            
            // 基于速度的降档
            if (speed_kmh < min_speed_thresholds[current_gear_]) {
                should_downshift = true;
            }
            
            // 低转速降档
            if (current_rpm < 850) {  // 从900调整为850
                should_downshift = true;
            }
            
            // 急加速降档（Kickdown）
            if (throttle > 0.8 && acceleration < 1.0) {
                should_downshift = true;
            }
            
            if (should_downshift) {
                current_gear_--;
                // 降档时转速上升
                last_rpm_ *= 1.3;
            }
        }
        
        // 停车时自动回空挡
        if (speed_kmh < 0.5 && throttle < 0.1) {
            current_gear_ = 0;
            // 回空挡时降低转速
            last_rpm_ = 100.0;
        }
    }

private:
    const mjModel* m_;
    int car_body_id_;
    int motor_forward_id_;
    
    // 成员变量
    double last_speed_ms_;
    double fuel_level_;
    int current_gear_;
    double gear_ratios_[7];  // 0-6档传动比
    double last_rpm_;        // 上一次的转速，用于换挡决策
    double max_speed_kmh_;   // 最大速度 (km/h)
    double max_rpm_;         // 最大转速 (RPM)
    const double base_gear_ratio = 12.0; // 基础传动比
};

}  // namespace mjpc

#endif  // MJPC_TASKS_SIMPLE_CAR_DASHBOARD_H_

