# Drone Environment - Legacy Version Archive

This document archives legacy drone control environments developed for various tasks including pose tracking, range-based landing, and trajectory tracking.
[Github:robotx-2022](https://github.com/ARG-NCTU/robotx-2022)
---

## Environment Family Overview

| Family | Purpose | Simulation Type | Primary Sensor |
|--------|---------|----------------|----------------|
| **Drone Pose** | Direct pose navigation | ROS/Gazebo | Ground truth pose |
| **Drone Range Land** | Platform landing with obstacles | Pure Python | UWB ranges |
| **Drone Range Track** | Free-space trajectory tracking | Pure Python | UWB ranges |
| **Drone Range YZ** | 2D plane control (Y-Z only) | Hybrid | UWB ranges |

---

# 1. Drone Pose Environment

## Drone Pose V1 (drone_pose_v1.py)

### Overview
**Purpose**: Train drone to navigate to goal poses using direct ground truth pose observations in ROS/Gazebo simulation.

### Configuration
- **Action Space**: Box(4) - `[vx, vy, vz, vyaw]` in range [-1, 1]
- **Observation Space**: Box(8) - Flattened history of relative poses
- **Control Frequency**: 20 Hz
- **Episode Length**: Unlimited (truncates if drone exceeds 100m from origin)

### Architecture
```python
DronePoseV1GYM (Gymnasium Interface)
    └── DronePoseV1ROS (ROS Backend)
        ├── SJTUROSController (Drone control)
        ├── GazeboROSConnector (Physics)
        └── FixedQueue (Observation history)
```

### Observation Details
**Structure**: `[goal_range(6), history(obs_hist_len * obs_frame_len)]`
- Uses transformation matrices to compute relative pose
- Converts goal pose to drone's local frame
- Maintains observation history (default: 2 frames × 4 values = 8 total)

**Transformation**:
```python
tm_drone_to_target = [R_z(yaw) | t]
tm_goal_to_target = [R_z(goal_yaw) | t_goal]
tm_goal_to_drone = inv(tm_drone_to_target) @ tm_goal_to_target
```

### Reward Function
```python
reward = -1.0 * position_error - 1.0 * angle_error
reward *= 1e-2

# Where:
position_error = ||drone_pos - goal_pos||
angle_error = ||drone_yaw - goal_yaw||
```

**Commented out components** (available for experimentation):
- Normalized action penalties based on error magnitude
- Cosine similarity bonus

### Spawn Configuration
- **Drone**: Currently spawns at origin `[0, 0, 0, random_yaw]`
- **Goal**: Currently spawns at origin `[0, 0, 0, random_yaw]`
- Yaw randomized in range `[-π, π]`

### Curriculum Learning Support
**Built-in schedulers** (currently disabled):
```python
r_min_scheduler: 0.1 → 10.0 over 3M-8M steps
r_max_scheduler: 10.0 → 50.0 over 3M-8M steps
```
These can gradually increase spawn distance ranges during training.

### Termination Conditions
- **Truncated**: Drone position exceeds 100m from origin (penalty: -1000)
- **Terminated**: Never (continuous task)

### Key Features
- Full 6-DOF control in simulation
- Precise transformation-based relative pose computation
- History-based observations for velocity estimation
- Goal visualization via RViz markers

---

# 2. Drone Range Land Environment

## Overview
**Task**: Navigate and land on a platform while avoiding obstacles, using only UWB range measurements (no direct pose).

### Common Configuration (All Versions)
- **Action Space**: Box(4) - `[vx, vy, vz, vyaw]` scaled by max speeds
  - Max linear speed: 1.0 m/s
  - Max angular speed: 1.0 rad/s
  - Timestep: 0.1s
- **Observation Space**: Box(24) - `[goal_signature(12), current_signature(12)]`
- **Episode Length**: 128 steps
- **Simulation**: Pure Python kinematics (no physics engine)

### UWB Configuration
**Hardware layout from YAML files**:
- **Anchors**: 6 fixed positions on platform (wamv.yaml)
- **Tags**: 2 tags on drone body (drone.yaml)
- **Signature**: 2 tags × 6 anchors = 12 distances per observation
- **Normalization**: All distances divided by 50.0

### Obstacle Definition
```python
obstacle = {
    "x": [-1.9, 2.8],  # 4.7m wide
    "y": [-1.5, 1.5],  # 3.0m wide  
    "z": [-inf, 0.5]   # Platform surface at z=0.5m
}
```

### Spawn Configuration
**Goal**: Near platform surface
- X: [-0.2, 0.2] m
- Y: [-0.1, 0.1] m
- Z: [0.5, 1.5] m (0-1m above platform)
- Yaw: [-π, π] rad

**Drone**: Wide randomization
- X: [-5, 5] m
- Y: [-5, 5] m
- Z: [-1, 5] m
- Yaw: [-π, π] rad
- **Constraint**: Must spawn outside obstacle

---

## Drone Range Land V1

### Reward Function
```python
dist = dist_xyz + 0.5 * dist_yaw

reward = (
    + 2.0 * (last_dist - dist)              # Progress reward
    - dist / EPISODE_LEN                     # Distance penalty
    + min(10.0, 1.0 / dist) / EPISODE_LEN   # Proximity bonus
    - 2.0 if terminated else 0.0             # Crash penalty
)

# Above-platform bonus (within 2m altitude):
if above_platform_xy and (0.5 < z <= 2.5):
    reward += min(1.0 / (z - 0.5), 5.0) / EPISODE_LEN
```

### Termination
- **Only on collision**: `terminated = is_inside_obstacle(x, y, z)`
- No success condition (must complete 128 steps)

### Behavioral Characteristics
- Focuses on obstacle avoidance
- Encourages hovering above platform
- No explicit landing reward

---

## Drone Range Land V2

### Changes from V1
**Added success termination**:
```python
if dist_xyz < 0.2 and dist_yaw < 0.1745329:  # 0.2m, 10°
    reward += 20.0
    terminated = True
```

### Behavioral Impact
- Episodes can end early on successful landing
- Strong reward incentive for precision landing
- More efficient training (no wasted steps after success)

---

## Drone Range Land V3

### Major Addition: NLOS Simulation

**New Component**: `RectangleGrid` class for realistic UWB multipath modeling

#### Region-Based NLOS System
Divides space into 9 regions around platform:
```
Region layout (top view):
  1 | 2 | 3
  ---------
  4 | 9 | 5
  ---------
  6 | 7 | 8

Region 9: Inside platform boundaries (LOS to all)
Regions 1-8: Outside platform (NLOS to some anchors)
```

#### NLOS Mask Table
```python
# Rows: Regions 1-9
# Cols: Anchors 0-5
# 0 = LOS, 1 = NLOS
masks = [
    [0, 0, 1, 0, 1, 0],  # Region 1 (top-right)
    [0, 1, 1, 0, 1, 0],  # Region 2 (top)
    [0, 1, 0, 0, 1, 0],  # Region 3 (top-left)
    [0, 1, 0, 1, 1, 1],  # Region 4 (left)
    [0, 0, 0, 1, 0, 1],  # Region 5 (bottom-left)
    [1, 0, 0, 1, 0, 1],  # Region 6 (bottom)
    [1, 0, 0, 0, 0, 1],  # Region 7 (bottom-right)
    [1, 0, 1, 0, 1, 1],  # Region 8 (right)
    [0, 0, 0, 0, 0, 0],  # Region 9 (inside - all LOS)
]
```

#### NLOS Distance Model
**Empirical model from literature**:
```python
distances[NLOS] = 1.48 * distances[NLOS] + 0.289
```
This adds ~48% range bias plus 28.9cm offset for blocked signals.

#### Implementation
```python
def pose_to_signature(self, pose):
    uwb_tag_pos = self.body_pose_to_uwb_tag_pos(pose)
    regions = self.rectangles.determine_region(uwb_tag_pos[:, :2])
    nlos_masks = self.rectangles.regions_to_nlos_masks(regions).flatten()
    
    distances = np.linalg.norm(uwb_tag_pos[:, None, :] - self.uwb_anchor_pos, axis=2).flatten()
    distances[nlos_masks == 1] = 1.48 * distances[nlos_masks == 1] + 0.289
    signature = distances / 50.0
    return signature
```

### Removed from V2
- Success termination (back to V1 behavior)
- Must complete full 128 steps

### Behavioral Impact
- More realistic sensor noise
- Policy must be robust to biased range measurements
- Harder to learn precise localization

---

## Drone Range Land V4

### Configuration
**Combines best of V2 and V3**:
- NLOS simulation from V3
- Success termination from V2

### Complete Feature Set
✅ Obstacle avoidance
✅ NLOS-affected UWB measurements  
✅ Success termination on precise landing
✅ Realistic sensor modeling

### Reward Function
Same as V1/V2 base, with success bonus.

### Use Case
**Most realistic training environment** in the Land family:
- Real-world sensor characteristics
- Clear success/failure signals
- Efficient episode termination

---

# 3. Drone Range Track Environment

## Overview
**Task**: Free-space trajectory tracking using UWB ranges - no obstacles or landing requirement.

### Differences from Land Series
- **No obstacle**: Full 3D navigation space
- **Different spawn**: Randomized in ±10m range
- **Goal proximity**: Drone spawns within ±10m offset from goal
- **Focus**: Pure tracking accuracy, not landing

### Common Configuration (All Versions)
- **Action/Observation**: Same as Land series
- **Episode Length**: 128 steps
- **Spawn Range**: ±10m in XYZ
- **Initial Offset**: ±10m from goal position

---

## Drone Range Track V1

### Reward Function
```python
dist = dist_xyz + 0.5 * dist_yaw

reward = (
    + 2.0 * (last_dist - dist)              # Progress toward goal
    - dist / EPISODE_LEN                     # Distance penalty
    + min(10.0, 1.0 / dist) / EPISODE_LEN   # Proximity bonus
)
```

**Simpler than Land V1**:
- No obstacle penalties
- No above-platform bonuses
- Pure distance minimization

### Termination
- **None**: Always runs 128 steps
- Continuous tracking task

---

## Drone Range Track V2

### Addition: Success Termination
```python
if dist_xyz < 0.2 and dist_yaw < 0.1745329:  # 0.2m, 10°
    reward += 20.0
    terminated = True
```

### Behavioral Change
- Episodes end early when goal reached
- Encourages faster convergence
- More sample-efficient training

---

## Drone Range Track V3

### Addition: NLOS Simulation
- Same `RectangleGrid` implementation as Land V3
- NLOS model: `distances[NLOS] = 1.48 * distances[NLOS] + 0.289`

### Debug Output
```python
print(f"V3 Environment initialized: {seed}")
```
Note: Despite comment saying "V3", this helps identify NLOS-enabled versions.

### Termination
- **None**: Back to continuous tracking (like V1)

---

## Drone Range Track V4

### Configuration
**Complete feature set**:
- NLOS simulation from V3
- Success termination from V2

### Final Version Features
✅ Realistic UWB sensor modeling  
✅ Clear success criteria
✅ Efficient episode termination
✅ No obstacle constraints

### Use Case
Best for **open-space navigation testing** with realistic sensors.

---

# 4. Drone Range YZ Environment

## Drone Range YZ V1 (drone_range_yz_v1.py)

### Overview
**Specialized 2D control** - Restricts drone movement to Y-Z plane (X fixed at 0).

### Purpose
- Simplified learning problem (2D instead of 3D)
- Faster training due to reduced action space
- Good for initial policy development

### Two Implementations

#### 4.1 DroneRangeYZV1GYM (ROS/Gazebo)

**Full simulation variant**:
```python
Action Space: Box(2) - [vy, vz] only
Observation Space: Box(12) - [goal_range(6), history(6)]
```

**Architecture**:
```
DroneRangeYZV1GYM
    └── DroneRangeYZV1ROS
        ├── SJTUROSController
        ├── UWBROSConnector  
        ├── GazeboROSConnector
        └── FixedQueue (history=1 frame)
```

**Configuration**:
- Observation history: 1 frame (6 values)
- Goal range: 6 UWB distances
- Total observation: 12D

**Spawn Settings**:
```python
goal_yz_range = 5.0
Goal Y: [-5, 5] m
Goal Z: [-5, 5] m
Drone: Goal ± random offset

# Alternative grid-based spawn (commented out):
# step = 0.1
# Systematic grid sampling in Y-Z plane
```

**Reward Function**:
```python
dist = ||range - goal_range||
reward = exp(-dist²) / max_episode_steps  # Normalized by 512 steps
```

**Alternative rewards** (commented out):
- Cosine similarity: `dot(range_norm, goal_range_norm)`
- Direct pose distance: `exp(-||pose - goal_pose||²)`

**Movement Command**:
```python
# Always sends 4D command, but only Y-Z are non-zero
self.env_ros.move(0.0, action[0], action[1], 0.0)
```

**Reset Behavior**:
```python
# Sends multiple zero commands to stabilize
self.env_ros.move(0, 0, 0, 0)  # x4 times
```

**Termination**:
- **Truncated**: If range distance > 5m
- **Terminated**: Never

---

#### 4.2 DroneRangeYZSimpleV0 (Pure Python)

**Lightweight kinematic variant**:

**Key Differences from GYM version**:
- No ROS/Gazebo - pure Python simulation
- Simpler kinematics (no physics)
- Direct position updates

**Configuration**:
```python
Action Space: Box(2) - [vy, vz]
Observation Space: Box(12) - [goal_sig(6), current_sig(6)]
dt = 0.05s (vs 0.1s in other envs)
```

**Position Update**:
```python
dy, dz = action * self.dt
self.drone_pose[1] += dy  # Direct Y update
self.drone_pose[2] += dz  # Direct Z update
# No rotation, no dynamics
```

**Reward**:
```python
dist = ||signature - goal_signature||
reward = exp(-dist²) / 1024  # Fixed normalization
```

**Spawn Configuration**:
```python
respawn_range = 7.0  # Fixed at 7m
Goal Y, Z: [-7, 7] m
Drone: Goal ± [-3, 3] m offset
```

**Curriculum Learning** (commented out):
```python
# Progressive difficulty based on total steps:
ranges = [
    ((0, 2M/70), 2.0),      # Easy: ±2m
    ((2M/70, 7M/70), 5.0),  # Medium: ±5m
    ((7M/70, 15M/70), 7.0), # Hard: ±7m
    ((15M/70, 30M/70), 10.0)# Expert: ±10m
]
```

**UWB Signature Calculation**:
```python
# Different from other envs - uses minimum distance
distances = ||uwb_tag_pos[:, None, :] - uwb_anchor_pos||
signature = min(distances, axis=0)  # Min across tags
```
Note: Most other envs flatten all tag-anchor pairs.

**Counter Tracking**:
```python
self.counter_step          # Steps in current episode
self.counter_total_step    # Total steps across all episodes
self.counter_episode       # Episode count
```

---

### Comparison: GYM vs Simple

| Feature | GYM Version | Simple Version |
|---------|-------------|----------------|
| **Backend** | ROS/Gazebo | Pure Python |
| **Physics** | Full dynamics | Kinematic only |
| **Timestep** | 0.1s | 0.05s |
| **Observation** | 6 + 6 with history | 6 + 6 no history |
| **Reward norm** | 1/512 | 1/1024 |
| **Spawn range** | ±5m | ±7m |
| **UWB calc** | All pairs flattened | Min distance |
| **Termination** | Truncate if dist>5 | Never |
| **Speed** | Slower (ROS) | Faster (Python) |
| **Realism** | High | Low |
| **Use case** | Final validation | Rapid prototyping |

---

# Environment Family Comparison Matrix

## Feature Comparison

| Environment | Simulation | Obs Type | Action Dim | Obstacle | NLOS | Success Term | Episode Len |
|-------------|-----------|----------|------------|----------|------|--------------|-------------|
| **Pose V1** | ROS/Gazebo | Direct pose | 4D | No | N/A | No | Unlimited |
| **Land V1** | Python | UWB ranges | 4D | Yes | No | No | 128 |
| **Land V2** | Python | UWB ranges | 4D | Yes | No | **Yes** | ≤128 |
| **Land V3** | Python | UWB ranges | 4D | Yes | **Yes** | No | 128 |
| **Land V4** | Python | UWB ranges | 4D | Yes | **Yes** | **Yes** | ≤128 |
| **Track V1** | Python | UWB ranges | 4D | No | No | No | 128 |
| **Track V2** | Python | UWB ranges | 4D | No | No | **Yes** | ≤128 |
| **Track V3** | Python | UWB ranges | 4D | No | **Yes** | No | 128 |
| **Track V4** | Python | UWB ranges | 4D | No | **Yes** | **Yes** | ≤128 |
| **YZ V1 GYM** | ROS/Gazebo | UWB ranges | 2D | No | No | No | 512 |
| **YZ V1 Simple** | Python | UWB ranges | 2D | No | No | No | 1024 |

## Spawn Range Comparison

| Environment | Drone Spawn Range | Goal Spawn Range | Notes |
|-------------|-------------------|------------------|-------|
| **Pose V1** | [0, 0, 0, ±π] | [0, 0, 0, ±π] | Currently fixed at origin |
| **Land Series** | [-5, 5] × [-5, 5] × [-1, 5] | [-0.2, 0.2] × [-0.1, 0.1] × [0.5, 1.5] | Goal near platform |
| **Track Series** | Goal ± 10m offset | [-10, 10]³ | Wide randomization |
| **YZ GYM** | Goal ± 0.5m | [-5, 5]² in YZ | 2D plane only |
| **YZ Simple** | Goal ± 3m | [-7, 7]² in YZ | Larger range |

## Reward Scale Comparison

| Environment | Base Reward Scale | Max Bonus | Notes |
|-------------|-------------------|-----------|-------|
| **Pose V1** | ×0.01 | None | Simple distance-based |
| **Land V1-V4** | Various | +20.0 | Success bonus in V2/V4 |
| **Track V1-V4** | Various | +20.0 | Success bonus in V2/V4 |
| **YZ GYM** | ÷512 | None | Episode-normalized |
| **YZ Simple** | ÷1024 | None | Fixed normalization |

---

# Technical Implementation Details

## UWB Signature Calculation

### Standard Method (Most Environments)
```python
def pose_to_signature(self, pose: np.ndarray) -> np.ndarray:
    uwb_tag_pos = self.body_pose_to_uwb_tag_pos(pose)
    distances = np.linalg.norm(
        uwb_tag_pos[:, None, :] - self.uwb_anchor_pos, 
        axis=2
    )
    signature = distances.flatten() / 50.0
    return signature
```
**Result**: 2 tags × 6 anchors = 12 distances

### Alternative (YZ Simple)
```python
distances = np.linalg.norm(
    uwb_tag_pos[:, None, :] - self.uwb_anchor_pos[None, :, :], 
    axis=2
)
signature = np.min(distances, axis=0)
```
**Result**: 6 distances (min across tags)

## Body Frame Transformation

### 3D Transformation (4D envs)
```python
def body_pose_to_uwb_tag_pos(self, pose: np.ndarray) -> np.ndarray:
    x, y, z, yaw = pose
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw], 
        [sin_yaw, cos_yaw]
    ])
    
    uwb_tag_pos_2d = self.uwb_tag_pos[:, :2]
    tag_pos_transformed_2d = np.dot(uwb_tag_pos_2d, rotation_matrix.T) + np.array([x, y])
    
    tag_pos_transformed = np.empty_like(self.uwb_tag_pos)
    tag_pos_transformed[:, :2] = tag_pos_transformed_2d
    tag_pos_transformed[:, 2] = z  # Z is not rotated
    
    return tag_pos_transformed
```

### Homogeneous Transformation (YZ Simple)
```python
transformation_matrix = np.array([
    [cos_yaw, -sin_yaw, x], 
    [sin_yaw, cos_yaw, y], 
    [0, 0, 1]
])
uwb_tag_pos_homogeneous = np.hstack((
    self.uwb_tag_pos[:, :2], 
    np.ones((self.uwb_tag_pos.shape[0], 1))
))
tag_pos_transformed_2d = np.dot(uwb_tag_pos_homogeneous, transformation_matrix.T)
```

## Angle Normalization

**Standard implementation across all environments**:
```python
@staticmethod
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
```
Wraps angles to `[-π, π]` range.

---

# Training Recommendations

## Environment Selection Guide

### For Initial Development
**Recommended**: `DroneRangeYZSimpleV0`
- Fastest iteration
- Simplest problem (2D)
- No simulation overhead
- Good for algorithm debugging

### For Realistic Training
**Land Task**: `DroneRangeLandV4`
- Full feature set
- NLOS + success termination
- Most realistic sensors

**Track Task**: `DroneRangeTrackV4`
- Full feature set
- No obstacle constraints
- Good for pure navigation

### For Sim-to-Real Transfer
**Recommended**: `DronePoseV1` → `DroneRangeYZV1GYM` → Land/Track V4
1. Learn pose control in simulation
2. Transfer to 2D range-based control
3. Scale to 3D with NLOS

### For Curriculum Learning
**Progressive difficulty**:
1. `YZ Simple` - 2D kinematics
2. `YZ GYM` - 2D with physics
3. `Track V1` - 3D without NLOS
4. `Track V3` - 3D with NLOS
5. `Land V4` - Full task complexity

## Hyperparameter Recommendations

### Episode Length
- **Pose V1**: 512-1024 steps (adjust based on task)
- **Land/Track**: 128 steps (fixed, proven effective)
- **YZ**: 512-1024 steps (more steps for finer control)

### Reward Scaling
- All environments pre-scale rewards
- No additional normalization usually needed
- Consider clipping rewards to [-10, 10] range

### Success Criteria
Environments with success termination use:
- **Position**: < 0.2m error
- **Orientation**: < 10° error (~0.1745 rad)

Consider tightening for real deployment:
- Position: < 0.1m
- Orientation: < 5°

---

# Migration and Upgrade Paths

## From Land V1 → V4
```python
# V1: Basic
env = DroneRangeLandV1(seed=0)

# V2: Add success termination
env = DroneRangeLandV2(seed=0)

# V3: Add NLOS (removes success)
env = DroneRangeLandV3(seed=0)

# V4: Best of both
env = DroneRangeLandV4(seed=0)
```

## From Track V1 → V4
```python
# Same progression as Land series
# V1 → V2: success termination
# V2 → V3: NLOS (removes success)
# V3 → V4: NLOS + success
```

## From Simple → Full Simulation
```python
# Start with YZ Simple for rapid development
env = DroneRangeYZSimpleV0(seed=0)

# Transfer learned policy to GYM version
env = DroneRangeYZV1GYM(seed=0)

# Scale to 3D
env = DroneRangeTrackV4(seed=0)
```

---

# Known Issues and Limitations

## All Environments
- **UWB YAML paths**: Hard-coded to specific ROS workspace
  ```python
  ~/robotx-2022/catkin_ws/src/pozyx_ros/config/wamv/
  ```
  Update these paths for your installation.

## Pose V1
- Currently spawns at origin only (randomization commented out)
- Schedulers implemented but not activated
- No explicit success criteria

## Land/Track Series
- Python kinematics: No collision dynamics
- NLOS model assumes rectangular platform
- Success criteria might be too loose for precision tasks

## YZ Environments
- **GYM version**: Requires full ROS/Gazebo setup
- **Simple version**: No physics, unrealistic dynamics
- Different UWB calculations between versions

## NLOS Implementation
- Fixed 9-region grid, doesn't adapt to platform size
- Empirical NLOS model may not match real hardware
- No height-dependent NLOS effects

---

# File Organization Recommendations

```
environments/
├── legacy/
│   ├── drone_pose_v1.py
│   ├── drone_range_land_v1.py
│   ├── drone_range_land_v2.py
│   ├── drone_range_land_v3.py
│   ├── drone_range_land_v4.py
│   ├── drone_range_track_v1.py
│   ├── drone_range_track_v2.py
│   ├── drone_range_track_v3.py
│   ├── drone_range_track_v4.py
│   └── drone_range_yz_v1.py
├── current/
│   ├── uavlander_v5.py  # Current main environment
│   └── ...
└── docs/
    ├── LEGACY_ARCHIVE.md  # This document
    └── VERSION_HISTORY.md  # UAVLander versions
```

---

# Deprecation Notes

## Superseded By
Most legacy environments have been superseded by the **UAVLander** series which offers:
- Better modular architecture
- Curriculum learning support
- Render capabilities
- More consistent API

## When to Use Legacy Environments
- **Reproducing old experiments**: Use exact legacy version
- **Specific features**: Some environments have unique capabilities
  - YZ for 2D control
  - Pose for direct ground truth
  - NLOS modeling in Land/Track V3/V4
- **Educational purposes**: Simpler code, easier to understand

---

# Appendix: Code Snippets

## Creating NLOS-Enabled Environment
```python
from drone_range_land_v4 import DroneRangeLandV4

env = DroneRangeLandV4(seed=42)
obs, info = env.reset()

for step in range(128):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(f"Success at step {step}!")
        break
```

## Testing UWB Signature
```python
env = DroneRangeLandV4(seed=0)
env.reset()

# Test NLOS effect
test_pose = np.array([3.0, 0.0, 1.0, 0.0])  # Outside platform
signature_with_nlos = env.pose_to_signature(test_pose)

# Compare with LOS (would need to modify code)
# Shows ~48% bias for blocked signals
```

## Curriculum Training Example
```python
# Progressive difficulty
envs = [
    DroneRangeYZSimpleV0(seed=0),    # Stage 1: 2D simple
    DroneRangeTrackV1(seed=0),        # Stage 2: 3D basic
    DroneRangeTrackV3(seed=0),        # Stage 3: 3D + NLOS
    DroneRangeLandV4(seed=0),         # Stage 4: Full task
]

# Train progressively
for i, env in enumerate(envs):
    print(f"Training stage {i+1}...")
    # Your training loop here
    # Load policy from previous stage if i > 0
```

## Multi-Environment Wrapper
```python
import gymnasium as gym
from typing import Dict, Any

class MultiModalDroneEnv(gym.Env):
    """Wrapper supporting multiple observation modalities"""
    
    def __init__(self, mode='pose'):
        if mode == 'pose':
            self.env = DronePoseV1GYM(seed=0)
        elif mode == 'range':
            self.env = DroneRangeLandV4(seed=0)
        elif mode == 'yz':
            self.env = DroneRangeYZV1GYM(seed=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
```

---

# Appendix: Reward Function Analysis

## Reward Component Breakdown

### Progress-Based Reward
Used in: Land/Track series
```python
progress_reward = 2.0 * (last_dist - dist)
```
**Characteristics**:
- Dense reward signal
- Encourages consistent movement toward goal
- Can be positive or negative (depends on progress)
- Scale: typically [-2, 2] per step

### Distance Penalty
Used in: Land/Track series
```python
distance_penalty = -dist / EPISODE_LEN
```
**Characteristics**:
- Encourages staying close to goal
- Normalized by episode length
- Always negative or zero
- Scale: [-∞, 0] (unbounded, but typically [-0.5, 0])

### Proximity Bonus
Used in: Land/Track series
```python
proximity_bonus = min(10.0, 1.0 / dist) / EPISODE_LEN
```
**Characteristics**:
- Strong signal near goal
- Capped at 10.0 to prevent instability
- Normalized by episode length
- Scale: [0, 10/128] ≈ [0, 0.078]

### Exponential Reward
Used in: Pose V1, YZ series
```python
reward = exp(-dist²)
```
**Characteristics**:
- Very sparse at large distances
- Sharp gradient near goal
- Always positive
- Scale: (0, 1]

### Success Bonus
Used in: Land/Track V2, V4
```python
success_bonus = 20.0
```
**Characteristics**:
- One-time large reward
- Clear success signal
- Helps credit assignment
- Dominates episode return on success

# Final Notes

## When to Use These Environments

✅ **USE** legacy environments when:
- Reproducing published results
- Need specific features (2D control, NLOS modeling)
- Educational/learning purposes
- Quick prototyping with Python-only envs

❌ **DON'T USE** legacy environments when:
- Starting new projects (use UAVLander)
- Need rendering capabilities
- Want curriculum learning support
- Require active maintenance

