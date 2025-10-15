# UAV Lander Environment - Version History

## Version 1.0 (uavlander_v1.py)
**Initial Release - Basic Reward Shaping**

### Configuration
- **Max Episode Steps**: 512
- **UWB History Length**: 3 frames
- **UWB Frame Length**: 6
- **Action History Length**: 3 frames
- **Action Dimensions**: 5 (x, y, z, yaw, landing_flag)
- **Landing Count Threshold**: 6

### Features
- Basic height-based reward shaping
- Simple curve function for reward calculation
- Landing action controlled by `action[4]`
- Takeoff command when `action[4] <= 0`

### Observation Space
```python
{
    'uwb_ranges': (3, 6),
    'uav_action': (3, 5)
}
```

### Reward Components
- **Height Score**: Based on distance to goal height (2m above platform)
- **Landing Bonus**: Incremental reward when landing conditions met
- **Landing Penalty**: `-reward_max/5` for failed landing attempt
- **Timeout Penalty**: -100 for exceeding max steps
- **Water Touch Penalty**: `-10 * reward_max`

### Key Variables
- `self.prev_distance`: Tracked but not used in reward
- Goal tolerance: ±0.2m vertical, 0.3m horizontal

---

## Version 2.0 (uavlander_v2.py)
**Enhanced Reward Shaping with Action Penalties**

### Configuration Changes
- **Max Episode Steps**: 512 → **1024**
- **Landing Count Threshold**: 6 → **3**
- **Action History Length**: 3 → 3 (unchanged)

### New Features
- Action-based reward shaping
- Tracking of previous radius (`self.prev_r`) and height (`self.prev_h`)
- More sophisticated movement penalties/bonuses

### Reward Function Overhaul
**New reward components:**
```python
reward = hight_score * curve(r) + (
    - 5  * abs(vz) if moving up when above platform
    + 50 * abs(vz) if moving down when below platform
    - 1  * abs(yaw_action)
    - 5  * xy_velocity if moving away from center when above platform
    + 20 * curve(r) if moving toward center when above platform
    + 2  * xy_velocity if moving away when below platform
)
```

### Modified Penalties
- **Height Score**: Now penalizes being below platform: `-reward_max * curve(abs(h+goal_z))`
- **Failed Landing**: `-reward_max/5` (unchanged)
- **Timeout**: -100 (unchanged)

### Behavioral Changes
- Encourages aggressive descent when below platform
- Discourages upward movement when above platform
- Rewards horizontal convergence when above platform

---

## Version 3.0 (uavlander_v3.py)
**Extended History & Refined Reward Structure**

### Configuration Changes
- **UWB History Length**: 3 → **6 frames**
- **Action History Length**: 3 → **6 frames**
- **Landing Count Threshold**: 3 → **4**

### Observation Space Changes
```python
{
    'uwb_ranges': (6, 6),  # Extended history
    'uav_action': (6, 5)   # Extended history
}
```

### Reward Simplification
Consolidated complex action penalties into three main components:
```python
reward = dist_rew + vz_punish + radius_bonus

where:
- dist_rew = hight_score * curve(r * (1 - curve(h*0.5)))
- vz_punish = -10*vz*u(h+tol) + 20*vz*(1-u(h+tol))
- radius_bonus = 8*(prev_r - r)*u(h+tol)
```

### Modified Penalties
- **Failed Landing**: `-reward_max/5` → **`-reward_max`** (stricter)
- **Timeout**: -100 → **No penalty** (removed)
- **Success Threshold**: Changed to 4 consecutive successful steps

### New Features
- Added `success` flag for logging
- Debug output: `print("range: ", self.range[0])`
- Improved height-radius coupling in reward

### Behavioral Changes
- Height score now uses `curve(h)*u(h-r*2)` for better height-radius relationship
- More lenient on timeout, stricter on failed landing attempts
- Smoother convergence behavior

---

## Version 4.0 (uavlander_v4.py)
**Architectural Refactoring - Modular Design**

### Major Architectural Changes
**Separated ROS connectors into dedicated modules:**
- `SJTUROSController` → **`uavROSConnector`**
- `UWBROSConnector` → **Integrated into `uavROSConnector`**
- **New**: `wamvGazeboROSConnector` for platform
- **New**: `GazeboROSConnector` (separate module)

### Configuration Changes
- **UWB Frame Length**: 6 → **12** (doubled)
- **Action Dimensions**: 5 → **4** (removed landing action)

### Observation Space Changes
```python
{
    'uwb_ranges': (6, 12),  # Doubled frame length
    'uav_action': (6, 4)    # No landing action
}
```

### Removed Features
- Landing action flag (`action[4]`)
- Explicit landing command logic
- Conditional takeoff based on landing flag

### New Behavior
- **Automatic Landing Detection**: Always checks landing conditions
- Landing occurs when position criteria met, regardless of action
- Simplified action space: only x, y, z, yaw velocities

### Code Quality Improvements
- Cleaner module separation
- Better code organization
- Improved output formatting

### Access Pattern Changes
```python
# V3:
self.gazebo.uav_pose
self.gazebo.wamv_pose

# V4:
self.uav.gazebo_pose
self.wamv.gazebo_pose
```

---

## Version 5.0 (uavlander_v5.py)
**Curriculum Learning Ready - Direct Pose Observation**

### Major Paradigm Shift
**Observation modality change**: UWB ranges → **Direct pose observation**

### Observation Space Changes
```python
{
    # 'uwb_ranges': (6, 12),  # COMMENTED OUT
    'uav_action': (6, 4),
    'uav_gazebo_pose': (6, 4)  # NEW: Direct state observation
}
```

### New Features
**Rendering Support:**
- Added pygame-based visualization
- Render modes: `"human"`, `"rgb_array"`
- Display parameters: 600×400 viewport, 50 FPS
- Basic white canvas (placeholder for future visualization)

### Curriculum Learning Design
**Stage 1 (Current)**: Direct pose observation
```python
self.uav_pose_hist = CircularQueue(dim=(self.uav_act_hist_len, 4))
return {'uav_action': ..., 'uav_gazebo_pose': self.uav_pose_hist.get()}
```

**Stage 2 (Ready to activate)**: UWB-based observation
```python
# Uncomment these lines:
# self.uwb_hist = CircularQueue(dim=(self.uwb_hist_len, self.uwb_frame_len))
# return {'uwb_ranges': self.uwb_hist.get(), 'uav_action': ...}
```

### Removed Features
- Episode reward accumulation display
- Verbose reward component logging in terminal

### Model Naming
- Standardized to `'sjtu_drone'` in reset method

### Rendering Implementation
```python
metadata = {
    "render_modes": ["rgb_array", "human"],
    "render_fps": 50,
}
```

### Behavioral Changes
- Simplified reward output (removed detailed component logging)
- Focus on policy learning with privileged information
- Easy toggle between direct pose and sensor-based observation

---

## Version Comparison Matrix

| Feature | V1 | V2 | V3 | V4 | V5 |
|---------|----|----|----|----|-----|
| **Max Steps** | 512 | 1024 | 1024 | 1024 | 1024 |
| **UWB History** | 3 | 3 | 6 | 6 | 6* |
| **UWB Frame Len** | 6 | 6 | 6 | 12 | 12* |
| **Action History** | 3 | 3 | 6 | 6 | 6 |
| **Action Dims** | 5 | 5 | 5 | 4 | 4 |
| **Landing Threshold** | 6 | 3 | 4 | 4 | 4 |
| **Observation Type** | UWB | UWB | UWB | UWB | **Direct Pose** |
| **Architecture** | Monolithic | Monolithic | Monolithic | **Modular** | Modular |
| **Render Support** | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Landing Action** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **Curriculum Ready** | ✗ | ✗ | ✗ | ✗ | **✓** |

*Available but commented out for curriculum learning

---

## Reward Function Evolution

### V1: Basic Shaping
```
reward = height_score * (1-u(h)) + curve(abs(r)) * u(h)
```

### V2: Action-Based Penalties
```
reward = height_score * curve(r) + action_penalties
```
- Complex action-dependent terms
- 5+ conditional reward components

### V3: Simplified Components
```
reward = dist_rew + vz_punish + radius_bonus
```
- 3 main components
- Better height-radius coupling

### V4-V5: Unchanged Core
```
reward = hight_score * curve(r*(1-curve(h*0.5)))
       + vz_punish + radius_bonus
```
- Focus on architecture and observation changes

---

## Migration Guide

### V1 → V2
- Increase max steps to 1024
- Implement action tracking (`prev_r`, `prev_h`)
- Add action-based reward components

### V2 → V3
- Extend history buffers (3→6)
- Simplify reward function
- Remove timeout penalty
- Increase failed landing penalty

### V3 → V4
- Refactor connectors into separate modules
- Remove landing action from action space
- Update pose access patterns
- Double UWB frame length

### V4 → V5
**For Direct Pose (Current):**
- Add pose history tracking
- Replace UWB observation with direct pose
- Add render support
- Implement pygame visualization

**For UWB Curriculum (Future):**
- Uncomment UWB-related code
- Comment out direct pose observation
- Train with sensor-based observation

---

## Curriculum Learning Strategy

### Stage 1: Privileged Learning (V5 Current)
**Observation**: Direct Gazebo pose `(x, y, z, yaw)`
- Fast policy convergence
- Ground truth state information
- Easier credit assignment

### Stage 2: Sensor-Based Transfer (V5 Activation)
**Observation**: UWB ranges (12-dimensional)
- Realistic sensor modality
- Partial observability
- Transfer from Stage 1 policy

### Implementation Toggle
```python
# Stage 1 → Stage 2
# Comment out:
# 'uav_gazebo_pose': gym.spaces.Box(...)
# self.uav_pose_hist = ...

# Uncomment:
'uwb_ranges': gym.spaces.Box(...)
self.uwb_hist = ...
```

---

## Development Recommendations

### Next Version Ideas

**V6: Full Sensor Suite**
- IMU data integration
- Velocity observations
- Multi-modal sensor fusion

**V7: Dynamics Randomization**
- Variable wind conditions
- Platform motion simulation
- Mass/inertia variations

**V8: Multi-Agent**
- Multiple UAVs
- Cooperative landing
- Collision avoidance

**V9: Real-World Transfer**
- Sim-to-real gap reduction
- Actual UWB hardware integration
- Physical platform testing

---

## Notes

- All versions use the same reward scaling: `reward /= 1000`
- Goal height: 2m above platform bottom
- Goal tolerance: ±0.2m vertical, 0.3m horizontal radius
- Initial spawn: 4-6m height, ±2m XY randomization
- Yaw randomization: ±0.35 radians around 1.57 (90°)
- Physics properties set via Gazebo connector in all versions