# How We Got the Pencil to Balance

This document describes the development path from first simulation to a real pencil standing upright on the table. The technical stack is described in [project_overview.md](project_overview.md). This document focuses on *why* things are the way they are and what we learned the hard way.

---

## The Problem

We are balancing a pencil on its tip on a motorized 2-DOF table, controlled in real time using two asynchronous event cameras (DVS). The state we care about:

```
State(px, vx, ax, wx, py, vy, ay, wy)
```

The control input is just table position `(px_cmd, py_cmd)`. We observe only `(px, py, ax, ay)` — velocities have to be estimated.

The pencil angle increases roughly 10% per 10 ms under gravity. A standard 60 Hz camera is too slow. DVS cameras emit events asynchronously at the pixel level, giving us millisecond-scale latency on angle changes — which is why the sensor choice matters so much.

---

## The Sensing Pipeline: Why Hough Space Works

The core sensing algorithm is a recursive Hough-space line tracker ([src/system/sensor/algo/hough.py](../src/system/sensor/algo/hough.py)). Each camera independently tracks a line `(slope, intercept)` in pixel space. The two lines are then combined into a 3D pencil measurement.

The Hough tracker maintains a quadratic objective over line parameters:

```
J(m, b) = a·m² + cross_mb·m·b + c·b² + linear_m·m + linear_b·b
```

Each DVS event updates this objective, weighted by how close the event is to the current line estimate. Old evidence is forgotten at a rate controlled by `mixing_factor`. The current line estimate is just the minimum of this quadratic — a closed-form solve at every event.

**Why this works well:**

- No frame rate. The line estimate updates per event, not per frame.
- Graceful noise rejection: events far from the pencil get low weight.
- Forgetting: old edges from a past pencil position fade out naturally.
- Fast: the entire update is a few multiplies and an inversion of a 2×2 matrix.

The algorithm has two main knobs: `mixing_factor` (forgetting rate) and `inlier_stddev_px` (how tightly events must match the current line). Tuning these to the real hardware took iteration, but once they were right the tracker was reliable.

**The failure mode to avoid:** setting `mixing_factor` too high makes the tracker jumpy; too low and it won't follow a fast-moving pencil. We settled on ~0.01–0.02.

---

## Controllers: The Path to Delta LQR

We tried many controllers. The progression was roughly:

### 1. Pole Placement (starting point)

Classic full-state feedback. Place the discrete-time closed-loop poles, compute gain `K`, apply `u = u_ref - K(x - x_ref)`.

This worked in clean simulation but was too aggressive on the real table. The command signal was noisy and caused the servos to buzz. The table mechanics couldn't follow fast, jerky commands.

### 2. Smooth Pole Placement

Instead of commanding absolute position, augment the state with the previous command and solve for `Δu`. This forces the controller to think about *changes* to the command rather than absolute values, which naturally smooths the output.

This was a meaningful improvement. The table motion was smoother and the pencil stayed up longer. But steady-state position drift was a problem — a small asymmetry in the table or a slight sensor offset would cause the pencil to wander to one side over time.

### 3. LQR

Switching from pole placement to LQR gave us explicit cost-function control over the trade-off between position error, tilt error, and command effort. The `Q` and `R` matrices make tuning more principled than pole placement.

LQR alone had the same drift problem as pole placement.

### 4. Delta LQR (the winner)

[src/system/controller/delta_lqr.py](../src/system/controller/delta_lqr.py)

Delta LQR combines everything:

- **Augmented state:** `ξ = [x; u_prev]`, control input is `Δu`. This gives smooth commands structurally, not as a tuning trick.
- **LQR cost over `Δu`:** penalizes command changes, not just command magnitude.
- **Position reference integrator:** slowly moves the reference point to cancel steady-state offset:
  ```
  x_ref[px] -= pos_ref_ki · (px - px_ref_true) · dt
  ```
- **Tilt feedforward integrator:** accumulated position error feeds directly into an actuator bias, outside the LQR error signal:
  ```
  tilt_ff += tilt_ff_ki · (px - px_ref_true) · dt
  ```

The combination of the delta formulation with two slow integrators (one for position, one for tilt) is what finally gave us clean, sustained balancing. The delta formulation handles dynamics; the integrators absorb the slow drifts that no linear model captures perfectly.

**Why it's the smoothest:** The `Δu` formulation structurally penalizes jerky commands. Unlike a derivative term in PID, this is built into the cost function in a way that the LQR optimally trades off between correction speed and smoothness.

---

## State Estimation: Kalman vs LPF

We ran two estimators in parallel and blended them with a supervisor-controlled factor `est_k ∈ [0, 1]`:

- `est_k = 0` → trust LPF (low-pass filter + finite differences)
- `est_k = 1` → trust Kalman

### Why LPF for startup

The LPF estimator ([src/system/estimator/lpf.py](../src/system/estimator/lpf.py)) is simple and has no model. At startup, when the pencil is being held by hand, the true dynamics don't match the balancer model at all. A Kalman filter initialized with the balancer model would produce bad velocity estimates during this phase. The LPF just smooths measurements and finite-differences — it's robust to model mismatch.

### Why Kalman for balancing

The Kalman filter ([src/system/estimator/kalman.py](../src/system/estimator/kalman.py)) uses the full discrete-time linear plant model. When the pencil is near vertical and dynamics roughly match the model, the Kalman gives much better velocity estimates because it incorporates physical structure.

The important tuning insight: **increasing process noise on the velocity and angular velocity states made Kalman work much better on the real system.**

The linearized model is only an approximation. The table has friction, backlash, and nonlinear servo dynamics. Setting high process noise (`q_vel`, `q_ang_vel`) on the hidden states tells the filter "don't trust your velocity predictions too much; update aggressively from measurements." This made the Kalman competitive with the real hardware in a way it wasn't with the default noise settings.

The covariances are normalized by scale factors so the raw `q` values are interpretable:
```
Q[i,i] = q_param[i] / scale[i]²
```
where `pos_scale ≈ 2 mm`, `ang_scale ≈ 0.15°`, etc. This makes it easy to say "velocity uncertainty is 10× position uncertainty" without worrying about unit inconsistencies.

---

## The Sim-to-Real Problem: Why Calibration Was the Key

Simulation worked cleanly. The real system did not behave like the simulation. The gap came from two sources:

1. **Actuator nonlinearity:** the commanded table position was not the actual table position. Backlash, friction, and servo nonlinearity meant a given `px_cmd` produced a different `px_actual` depending on where you were in the workspace and which direction you were moving.

2. **Sensor cross-coupling:** the cameras measure a line in pixel space, not directly world position and tilt. Converting those pixel observations to world coordinates requires an inverse mapping that depends on the camera placement, lens distortion, and — critically — position and tilt are **not independent** in pixel space. The pixel intercept of the line depends on both where the pencil base is and how much the pencil is tilted, in a way that changes across the workspace.

Without solving both of these, the controller was fighting a completely wrong model of what was happening.

---

## Calibration: How It Actually Works

### Actuator Calibration

The actuator calibration answers: "if I send command `(px_cmd, py_cmd)`, what position does the table actually go to?"

**Collection:** We commanded the table to a grid of positions (4 cm steps across the 7 cm workspace, ~9–37 points) and used the camera to measure where the table actually ended up.

**Model:** Two Thin-Plate Spline (TPS) RBF interpolants:
```
rbf_px_actual: (px_cmd, py_cmd) → px_actual
rbf_py_actual: (px_cmd, py_cmd) → py_actual
```

TPS / RBF interpolation is exact at the training points (zero residual at collected samples) and smooth between them. This is important because it means the calibration is self-consistent — the measured points are reproduced exactly.

**At runtime:** We want to send the command that *achieves* a desired position, not the command that merely requests it. So we invert the learned forward map via Newton iteration: find `u_cmd` such that `rbf(u_cmd) = u_desired`.

The actuator calibration is clamped strictly to the calibrated region. Extrapolation of servo behavior outside the workspace is not reliable.

### Sensor Calibration: The Hard Part

The sensor calibration is more complex because of camera physics. The two cameras observe the pencil as a 2D line in pixel space. The observed line parameters — slope `s` and intercept `b` — depend on **both** position and tilt simultaneously, and this coupling varies across the workspace.

**Why coupling exists:** Consider a camera looking at a pencil from the side. If the pencil tilts forward, the line in the image rotates (slope changes) *and* the apparent intercept shifts (because the tip and base move in different directions). If the pencil is off-center, the projection is distorted differently than at center. So the mapping:

```
(px, py, ax, ay) → (s_cam1, b_cam1, s_cam2, b_cam2)
```

is genuinely 4D-to-4D and nonlinear. Treating it as independent per-axis linear maps (our v1 approach) fails at workspace corners where distortion is largest.

**Calibration evolution:**

**v1: Affine per-camera**
- `px = k1·b1 + c1`, `py = k2·b2 + c2`
- Tilt from slope with a simple linear lookup per camera
- Works near center, degrades toward corners

**v2: Bilinear with cross-coupling correction**
- `px = a0 + a1·b1 + a2·b2` — cross-term `a2` corrects first-order perspective
- Better, but still polynomial in a relationship that is fundamentally rational (projective geometry)

**v3: Thin-Plate Spline (current)**

The v3 calibration ([src/system/sensor/observation_model/simple_dvs_regression_model.py](../src/system/sensor/observation_model/simple_dvs_regression_model.py)) uses RBF interpolation throughout.

**Position calibration:**

We drove the table to a grid of known positions (2 cm steps in the 7 cm workspace, ~37 points) with the pencil held vertical, and recorded what pixel intercepts each camera reported:

```
rbf_b1: (px_m, py_m) → b1_px   (cam1 pixel intercept at this world position)
rbf_b2: (px_m, py_m) → b2_px   (cam2 pixel intercept at this world position)
```

At runtime, invert this via Newton iteration: find `(px, py)` such that the interpolants match the observed intercepts.

**Tilt calibration — the position-dependent part:**

With the table at known positions, we physically tilted the pencil to known angles and recorded what pixel *slope* each camera reported. This was done at multiple positions (center + corners of a cross pattern) to capture how the slope-to-angle mapping changes across the workspace:

```
rbf_s1: (px_m, py_m, α_x_rad) → s_px_cam1
rbf_s2: (px_m, py_m, α_y_rad) → s_px_cam2
```

So the same pixel slope corresponds to a *different* physical tilt depending on where the table is. Without position-dependent tilt calibration, we were systematically misreading the tilt at workspace edges, which confused the controller and destabilized the pencil.

At runtime, we extract tilt from slope via a 1D root-find (Brentq): find `α` such that `rbf_s(px, py, α) = s_observed`.

**Why TPS / RBF specifically:**

- Exact at training points — sanity check is built in
- Smooth interpolation between points — no discontinuities
- Works for arbitrary scattered data — no grid alignment required
- Generalizes well with enough calibration density
- Thin-plate splines minimize the "bending energy" of the interpolating surface — they're the smoothest possible interpolant for this kind of data

**Calibration data storage:**

All calibration data is stored as JSON with explicit versioning:
```json
{
  "version": "v3_tps",
  "positions_m": [...],
  "b1_px": [...],
  "tilt_positions_m": [...],
  "tilt_alphas_rad": [...],
  "tilt_s_px_cam1": [...]
}
```

The version field let us roll forward through v1 → v2 → v3 without breaking existing scripts.

---

## The Density Question: How Many Calibration Points Do You Need?

More is better, but the relationship is not linear. The key insight is that TPS/RBF interpolation is exact at training points and smooth between them — so the question is really about how fast the true mapping curves between samples.

For the actuator (servo behavior), the nonlinearity is fairly smooth, so a coarse grid (4 cm steps, ~9 points) is enough to get meaningful correction.

For the sensor tilt calibration, the position-dependence of the slope-to-angle map was sharper, especially near corners. We needed at least a 5-point cross (center + 4 offset positions) to capture it. Fewer than that and the interpolant would pass through the training points exactly but oscillate badly in between.

The practical test: drive to a calibration point, check that the reconstructed measurement matches the known ground truth. If it does, TPS is doing its job. Then check midpoints — if those are also consistent, density is sufficient.

---

## Why It All Clicked Together

The final working system is:

- **Sensing:** Two DVS cameras → Hough-space line tracking (per event, ~ms latency) → v3 TPS calibration (position-dependent, cross-coupled correction) → `(px, py, ax, ay)`
- **Estimation:** LPF at startup, blend to Kalman once the pencil is upright. Kalman works because model uncertainty is set high enough to allow aggressive measurement updates.
- **Control:** Delta LQR with position and tilt integrators. Smooth by construction, drift-free in steady state.
- **Actuation:** TPS-corrected inverse map from desired position to servo command.

Each piece had predecessors that worked in simulation or at low fidelity. The sim-to-real gap came almost entirely from:

1. The actuator doing something different from what it was told (solved by actuator calibration)
2. The sensor reading tilt incorrectly at off-center positions (solved by position-dependent tilt calibration)

Once both calibrations were dense enough and the v3 TPS model was in place, the system that worked in simulation started working in reality.

The controllers and estimators that came before — pole placement, smooth pole, LPF-only — were not wasted. They were stepping stones that built intuition for what the system needed, and they revealed exactly where the model mismatch was largest. But none of them would have worked on real hardware without calibration giving the sensor and actuator the accuracy the control theory assumes.
