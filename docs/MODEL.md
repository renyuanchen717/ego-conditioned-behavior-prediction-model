# Recorded K=12: exact workflow

## 1. Frozen foundation

The external NDPNet consumes scene history and map information and supplies six trajectory means X_ik(t) and their probabilities q_ik. The released residual model takes these predictions, not raw camera/LiDAR data or NDPNet embeddings. NDPNet parameters remain unchanged.

All geometry is expressed in the target agent's current position/heading frame. The inference grid is t = 0.5, 1.0, ..., 8.0 seconds.

## 2. Constant-velocity ego reference

$$b(t)=p_e(0)+v_e(0)t.$$

The recorded ego plan r(t) is used in offline diagnostics. In online planning, r(t) is a Flow Planner candidate, never an unavailable future log. WOMD diagnostics use the ego track's current velocity. The nuPlan integration estimates center velocity from the last two ego states; these are different velocity-source implementations and are documented rather than silently conflated.

## 3. Context and anchor

$$c_{ik}=\phi(X_i,q_i,\mathrm{type}_i,b,b),\qquad A_{ik}(t)=\mathrm{Head}(c_{ik})(t).$$

`features` produces 64 features per mode. They include probabilities, one-hot object type, target positions and ego-relative positions at 1/3/5/8 seconds, velocity/deviation features, and synchronous/asynchronous proximity features. In anchor context both plan inputs are CV, so their deviation features are zero.

The anchor MLP is 64 → 96 → LayerNorm → SiLU → 64 → SiLU → 8. It produces four 2D residual knots at 2/4/6/8s, bounded with 4*tanh(z/4), and interpolates them linearly with a zero residual at t=0.

$$R_{ik}(t)=X_{ik}(t)+A_{ik}(t).$$

## 4. Conditional response

For each native mode, the geometry functions compare ego and target paths via soft spatial conflict, relative arrival order and synchronous proximity:

$$s^5_{ik}=G_5(X_{ik},r)-G_5(X_{ik},b),\quad
s^8_{ik}=G_8(X_{ik},r)-G_8(X_{ik},b).$$

G5 uses times through 5s, split into two windows. G8 uses the full 8s with two four-second windows. Each returns six descriptors. The soft spatial weighting temperature is 1.5m and the proximity decay scale is 8m; the code contains the exact equations and index conventions.

Each response head maps context 64 → 48 → SiLU → 12, reshapes the result into a 2×6 matrix, applies it to clipped signal/8, bounds the result with 2*tanh, and rotates it from the mode's longitudinal/lateral basis into the target-local frame. A validation-selected scalar multiplies each head output:

$$d^h_{ik}=\alpha_h\,\mathrm{Response}_h(c_{ik},s^h_{ik},B_{ik}),\qquad h\in\{5,8\}.$$

When r=b, both conditional responses are exactly zero. The anchor residual is not necessarily zero.

## 5. Full-horizon proposals

Let S(u)=u²(3−2u), h5(t)=S(min(t/5,1)), and h8(t)=S(max((t−5)/3,0)). For the modeled 0–8s horizon:

$$P_{ik}(t)=R_{ik}(t)+h_5(t)[1-h_8(t)]d^5_{ik}+h_8(t)d^8_{ik}.$$

Thus P(5s)=R(5s)+d5 and P(8s)=R(8s)+d8.

$$\mathcal P_i=\{X_{i1},\ldots,X_{i6},P_{i1},\ldots,P_{i6}\}.$$

These are residual proposals attached to native modes, not guaranteed distinct maneuver categories. There is no mode deduplication, impact classification, probability recalibration, cross-agent joint decoder or K=6 compression in this locked version.

## 6. 10 Hz planning interface

The model computes residuals at 2 Hz. It inserts the zero residual at t=0, linearly interpolates residuals to 10 Hz, and adds them to the original 10 Hz NDPNet trajectories. The original six trajectories remain exactly unchanged, including their between-knot geometry.
