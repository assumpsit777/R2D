# R2D 弱形式推导与代码对应检查

本文档对应文件：

- `E:\Fenicsx\R2d\r2d_documented_model_strong_dofs.py`

记号约定：

- 物理坐标为 \(x\)，代码中网格坐标为 \(\hat x=x/L\)，其中 \(L=\texttt{p.length_scale}\)。
- 体积分 `dx(...)` 是无量纲区域上的面积积分，物理面积满足 \(d\Omega=L^2 d\hat\Omega\)。
- 边界/界面积分 `ds(...)`、`dS(...)` 是无量纲长度积分，物理长度满足 \(d\Gamma=L d\hat\Gamma\)。
- 论文中 \(\Gamma_s\) 上的法向 \(n\) 定义为相对于 \(\Omega_2\) 的外法向，即从 \(\Omega_2\) 指向 \(\Omega_3\)。
- 代码里整体残差都写成 \(F=0\)。

## 1. 正极界面 Butler-Volmer 反应

论文定义：

\[
\eta_c=\phi_s-\phi_l-E_{eq}(c)-\frac{\Omega_c\sigma_h}{F}
\]

\[
i_s=i_{0,c}\left[
\exp\left(-\frac{\alpha F\eta_c}{RT}\right)
-
\exp\left(\frac{(1-\alpha)F\eta_c}{RT}\right)
\right]
\]

\[
i_{0,c}
=i_{0,c}^{ref}
\left(2\theta\right)^\alpha
\left(2(1-\theta)\right)^{1-\alpha},
\qquad
\theta=\frac{c_{Li^+}}{c_{Li^+,max}}
\]

代码中 \(c\) 已经是归一化浓度 \(\theta\)，所以直接使用 `c`。

代码对应：

```python
theta = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, c))
e_eq = eeq_from_soc(theta)
e_eq_eff = overp_mech_c_volts
i0_c = p.i0_c_ref * (2.0 * theta) ** p.alpha * (2.0 * (1.0 - theta)) ** (
    1.0 - p.alpha
)
overp_c_volts = phis - phil - e_eq - e_eq_eff
overp_c = overp_c_volts / p.Vt
i_cathode = i0_c * (
    safe_exp(-p.alpha * overp_c, p) - safe_exp((1.0 - p.alpha) * overp_c, p)
)
```

界面上代码使用平均值：

```python
theta_s = ufl.max_value(theta_eps, ufl.min_value(1.0 - theta_eps, ufl.avg(c)))
overp_c_volts_s = (
    ufl.avg(phis)
    - ufl.avg(phil)
    - ufl.avg(e_eq)
    - ufl.avg(overp_mech_c_volts)
)
i_cathode_s = i0_c_s * (
    safe_exp(-p.alpha * overp_c_s, p) - safe_exp((1.0 - p.alpha) * overp_c_s, p)
)
```

检查点：

- 论文定义下 \(i_s>0\) 表示 Li 从 \(\Omega_2\) 进入 \(\Omega_3\)，即正极嵌锂。
- 充电时正极失锂，所以通常应有 \(i_s<0\)。
- 代码目前用 `avg(...)` 取界面值。由于 `phis` 只在 \(\Omega_2\) 有物理意义、`c` 只在 \(\Omega_3\) 有物理意义，真正 restricted space 版本里更干净的写法应使用指定侧 trace，而不是 `avg`。

## 2. \(\phi_l\) 离子电势方程

### 2.1 强式

在 \(\Omega_1\) 中，金属锂相场变化产生离子电流源项。代码采用：

\[
J_{cell}=\frac{F}{\Omega_{Li}}\frac{\xi^{n+1}-\xi^n}{\Delta t}
\]

论文符号按你前面确认的方向应写成：

\[
\nabla\cdot(-\sigma_{eff}\nabla\phi_l)=J_{cell}
\qquad \text{in } \Omega_1
\]

在 \(\Omega_2\) 中：

\[
\nabla\cdot(-\sigma_{SSE}\nabla\phi_l)=0
\qquad \text{in } \Omega_2
\]

\(\Gamma_s\) 上使用论文的 \(\Omega_2\) 外法向 \(n\)：

\[
-\sigma_{SSE}\nabla\phi_l\cdot n=-i_s
\qquad \text{on } \Gamma_s
\]

其他外边界自然绝缘：

\[
-\sigma\nabla\phi_l\cdot n=0
\]

### 2.2 弱式推导

对 \(\Omega_1\)：

\[
\int_{\Omega_1}\nabla\cdot(-\sigma_{eff}\nabla\phi_l)v_l\,d\Omega
=
\int_{\Omega_1}J_{cell}v_l\,d\Omega
\]

分部积分：

\[
\int_{\Omega_1}\sigma_{eff}\nabla\phi_l\cdot\nabla v_l\,d\Omega
-\int_{\partial\Omega_1}\sigma_{eff}\nabla\phi_l\cdot n\,v_l\,d\Gamma
-\int_{\Omega_1}J_{cell}v_l\,d\Omega=0
\]

对 \(\Omega_2\) 同理，并把 \(\Gamma_s\) 边界条件代入：

\[
\int_{\Omega_2}\sigma_{SSE}\nabla\phi_l\cdot\nabla v_l\,d\Omega
-\int_{\Gamma_s}i_s v_l\,d\Gamma=0
\]

将电势方程整体除以 \(L\) 后，用无量纲坐标写成代码形式：

\[
\int_{\hat\Omega_1}\frac{\sigma_{eff}}{L}\hat\nabla\phi_l\cdot\hat\nabla v_l\,d\hat\Omega
+
\int_{\hat\Omega_2}\frac{\sigma_{SSE}}{L}\hat\nabla\phi_l\cdot\hat\nabla v_l\,d\hat\Omega
-
\int_{\hat\Omega_1}LJ_{cell}v_l\,d\hat\Omega
-
\int_{\hat\Gamma_s}i_s v_l\,d\hat\Gamma
=0
\]

### 2.3 代码对应

```python
liion_source_expr = p.F * (1.0 / p.omega_li) * xit_expr
```

```python
F_phil = (
    (sigma_eff / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx(OMEGA1)
    + (p.sse / p.length_scale) * ufl.dot(ufl.grad(phil), ufl.grad(v_l)) * dx(OMEGA2)
    - p.length_scale * liion_source_expr * v_l * dx(OMEGA1)
    + cathode_phil_term
)
```

其中：

```python
phil_term = -i_s * ufl.avg(v_l) * dS(GAMMA_S)
```

所以代码实际为：

\[
F_{\phi_l}
=
\int_{\hat\Omega_1}\frac{\sigma_{eff}}{L}\hat\nabla\phi_l\cdot\hat\nabla v_l
+
\int_{\hat\Omega_2}\frac{\sigma_{SSE}}{L}\hat\nabla\phi_l\cdot\hat\nabla v_l
-
\int_{\hat\Omega_1}LJ_{cell}v_l
-
\int_{\hat\Gamma_s}i_s \operatorname{avg}(v_l)
\]

检查点：

- `- p.length_scale * liion_source_expr` 对应强式右端 \(+J_{cell}\) 移到残差左边。
- `phil_term = -i_s ...` 与 \(-\sigma\nabla\phi_l\cdot n=-i_s\) 一致。
- `p.sse` 是 \(\Omega_2\) 中固态电解质/混合区离子电导，符号上比旧的 `sigma_mix` 更接近论文。

## 3. \(\phi_s\) 电子电势方程

### 3.1 强式

\[
\nabla\cdot(-\sigma_s\nabla\phi_s)=0
\qquad \text{in } \Omega_2
\]

\(\Gamma_s\) 上：

\[
-\sigma_s\nabla\phi_s\cdot n=i_s
\qquad \text{on } \Gamma_s
\]

集流体 \(\Gamma_c\) 上代码写入外加电流 \(i_{app}\)：

\[
-\sigma_s\nabla\phi_s\cdot n=-i_{app}
\qquad \text{on } \Gamma_c
\]

注意这里 \(i_{app}\) 的正负是代码约定。当前代码充电阶段使用：

```python
charge_current_sign: float = 1.0
...
phases.append(("charge", p.charge_time, p.charge_current_sign * p.current_abs))
assign_constant(i_app, current_value)
```

所以默认充电时 `i_app=+10 A/m^2`。

### 3.2 弱式推导

\[
\int_{\Omega_2}\sigma_s\nabla\phi_s\cdot\nabla v_s\,d\Omega
-\int_{\partial\Omega_2}\sigma_s\nabla\phi_s\cdot n\,v_s\,d\Gamma=0
\]

代入边界通量：

\[
\int_{\Omega_2}\sigma_s\nabla\phi_s\cdot\nabla v_s\,d\Omega
+
\int_{\Gamma_s}i_s v_s\,d\Gamma
+
\int_{\Gamma_c}i_{app}v_s\,d\Gamma=0
\]

整体除以 \(L\) 后：

\[
\int_{\hat\Omega_2}\frac{\sigma_s}{L}\hat\nabla\phi_s\cdot\hat\nabla v_s\,d\hat\Omega
+
\int_{\hat\Gamma_s}i_s v_s\,d\hat\Gamma
+
\int_{\hat\Gamma_c}i_{app}v_s\,d\hat\Gamma=0
\]

### 3.3 代码对应

```python
F_phis = (
    (p.sigma_cathode / p.length_scale) * ufl.dot(ufl.grad(phis), ufl.grad(v_s)) * dx(OMEGA2)
    + cathode_phis_term
    + i_app * v_s * ds(GAMMA_C)
)
```

其中：

```python
phis_term = i_s * ufl.avg(v_s) * dS(GAMMA_S)
```

检查点：

- 这部分与
  \[
  \int_{\Omega_2}\sigma_s\nabla\phi_s\cdot\nabla v_s
  +\int_{\Gamma_s}i_s v_s
  +\int_{\Gamma_c}i_{app}v_s=0
  \]
  一致。
- 若充电时希望正极失锂，则在整体守恒上应看到
  \[
  \int_{\Gamma_s} i_s\,d\Gamma + \int_{\Gamma_c} i_{app}\,d\Gamma \approx 0
  \]
  因而 `i_app>0` 时通常 `Gs<0`。

## 4. 正极颗粒浓度 \(c\) 方程

### 4.1 强式

代码中 \(c\) 是归一化浓度：

\[
c=\frac{c_{Li^+}}{c_{Li^+,max}}
\]

论文通量：

\[
\mathbf J
=
-D\nabla c_{Li^+}
+
\frac{D c_{Li^+}\Omega_c}{RT}\nabla\sigma_h
\]

换成归一化 \(c\) 后：

\[
\frac{\partial c}{\partial t}
=-\nabla\cdot\mathbf j_c
\]

其中归一化通量可理解为：

\[
\mathbf j_c
=
-D\nabla c
+
\frac{D c\Omega_c}{RT}\nabla\sigma_h
\]

论文边界条件用 \(\Omega_2\) 外法向 \(n\)：

\[
\mathbf J\cdot n=\frac{i_s}{F}
\qquad \text{on } \Gamma_s
\]

但是 \(c\) 方程在 \(\Omega_3\) 上积分时，\(\Omega_3\) 的外法向为

\[
n_3=-n
\]

所以对 \(\Omega_3\) 的外法向：

\[
\mathbf J\cdot n_3=-\frac{i_s}{F}
\]

归一化后：

\[
\mathbf j_c\cdot n_3=-\frac{i_s}{F c_{max}}
\]

### 4.2 弱式推导

从

\[
\frac{\partial c}{\partial t}
=-\nabla\cdot\mathbf j_c
\]

得到：

\[
\int_{\Omega_3}\frac{\partial c}{\partial t}v_c\,d\Omega
+
\int_{\Omega_3}\nabla\cdot\mathbf j_c\,v_c\,d\Omega=0
\]

分部积分：

\[
\int_{\Omega_3}\frac{\partial c}{\partial t}v_c\,d\Omega
-
\int_{\Omega_3}\mathbf j_c\cdot\nabla v_c\,d\Omega
+
\int_{\partial\Omega_3}\mathbf j_c\cdot n_3\,v_c\,d\Gamma=0
\]

代入

\[
\mathbf j_c=-D\nabla c+\frac{Dc\Omega_c}{RT}\nabla\sigma_h
\]

得到：

\[
\int_{\Omega_3}c_t v_c\,d\Omega
+
\int_{\Omega_3}D\nabla c\cdot\nabla v_c\,d\Omega
-
\int_{\Omega_3}\frac{Dc\Omega_c}{RT}\nabla\sigma_h\cdot\nabla v_c\,d\Omega
-
\int_{\Gamma_s}\frac{i_s}{F c_{max}}v_c\,d\Gamma=0
\]

无量纲坐标下，整式除以 \(L^2\)：

\[
\int_{\hat\Omega_3}c_t v_c\,d\hat\Omega
+
\int_{\hat\Omega_3}\frac{D}{L^2}\hat\nabla c\cdot\hat\nabla v_c\,d\hat\Omega
-
\int_{\hat\Omega_3}\frac{D c\Omega_c}{RT L^2}\hat\nabla\sigma_h\cdot\hat\nabla v_c\,d\hat\Omega
-
\int_{\hat\Gamma_s}\frac{i_s}{F c_{max}L}v_c\,d\hat\Gamma=0
\]

### 4.3 代码对应

```python
F_c = (
    (c - c_n) / dt_const * v_c * dx(OMEGA3)
    + (p.D_li / p.length_scale**2)
    * ufl.dot(ufl.grad(c), ufl.grad(v_c))
    * dx(OMEGA3)
    - (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
    * c
    * ufl.dot(ufl.grad(hydro3), ufl.grad(v_c))
    * dx(OMEGA3)
    + cathode_c_term
)
```

其中：

```python
scale_c = 1.0 / (p.F * p.c_li_max * p.length_scale)
c_term = -scale_c * i_s * ufl.avg(v_c) * dS(GAMMA_S)
```

检查点：

- `c_term` 的负号来自 \(n_3=-n_{\Omega_2}\)。
- 如果 \(i_s<0\)，则 `c_term` 为正，表示正极失锂时 \(c_t\) 应趋向负值来抵消残差。
- 代码体项量纲缩放和“整式除以 \(L^2\)”一致。

## 5. Li 金属相场 \(\xi\) 的 Allen-Cahn 方程

### 5.1 强式

代码实际对应形式：

\[
\frac{1}{L_\sigma}\frac{\xi^{n+1}-\xi^n}{\Delta t}
-
\frac{\kappa}{L^2}\hat\Delta\xi
+
W\frac{\partial f}{\partial \xi}
+
\frac{\partial f_{mech}}{\partial\xi}
+
W_{gb,\xi}\xi\sum_i\eta_i^2
-
\frac{L_\eta}{L_\sigma}R_{\xi}
=0
\]

其中：

\[
f(\xi)=\xi^2(1-\xi)^2
\]

\[
R_{\xi}
=
\texttt{xi\_source\_weight}\cdot\texttt{deposition\_drive}
\]

代码中负极 BV：

\[
R_{Li}
=
\exp\left(\frac{(1-\alpha)F\eta_a}{RT}\right)
-
c_e\exp\left(-\frac{\alpha F\eta_a}{RT}\right)
\]

\[
\texttt{deposition\_drive}=-R_{Li}
\]

因此负过电势充电沉积时，`deposition_drive` 倾向为正，使 \(\xi\) 增大。

### 5.2 弱式

乘以测试函数 \(v_\xi\)，梯度项分部积分并采用自然边界：

\[
\int_{\Omega_1}
\frac{\xi^{n+1}-\xi^n}{\Delta t\,L_\sigma}v_\xi\,d\Omega
+
\int_{\Omega_1}
\frac{\kappa}{L^2}\hat\nabla\xi\cdot\hat\nabla v_\xi\,d\Omega
+
\int_{\Omega_1}
W f'(\xi)v_\xi\,d\Omega
+
\int_{\Omega_1}
\frac{\partial f_{mech}}{\partial\xi}v_\xi\,d\Omega
\]

\[
+
\int_{\Omega_1}
W_{gb,\xi}\xi\sum_i\eta_i^2 v_\xi\,d\Omega
-
\int_{\Omega_1}
\frac{L_\eta}{L_\sigma}R_\xi v_\xi\,d\Omega
=0
\]

### 5.3 代码对应

```python
F_xi = (
    (xi - xi_n) / dt_const / p.L_sigma * v_xi * dx(OMEGA1)
    + (p.kappa0 / p.length_scale**2)
    * ufl.dot(ufl.grad(xi), ufl.grad(v_xi))
    * dx(OMEGA1)
    + p.W_b * df_dxi * v_xi * dx(OMEGA1)
    + dfmech_dxi * v_xi * dx(OMEGA1)
    + p.W_gb_xi * xi * sum_eta_sq * v_xi * dx(OMEGA1)
    - p.xi_mobility_scale
    * (p.L_eta / p.L_sigma)
    * localized_reaction_li
    * v_xi
    * dx(OMEGA1)
)
```

负极反应部分：

```python
overp_li_volts = phil - p.e_eq_li - overp_mech_volts
overp_li = overp_li_volts / p.Vt
reaction_li_raw = safe_exp((1.0 - p.alpha) * overp_li, p) - ufl.min_value(
    ce_expr, 1.0
) * safe_exp(-p.alpha * overp_li, p)
reaction_li = safe_reaction(reaction_li_raw, p)
deposition_drive = -reaction_li
xi_source_weight = hp(xi) + p.gb_xi_source_scale * B_expr * (1.0 - hxi)
localized_reaction_li = xi_source_weight * deposition_drive
```

检查点：

- `reaction_li` 不是正极的 \(i_s\)，它是负极 Li/Li+ BV 的无量纲括号项，不要和 `i_cathode_s` 混。
- `localized_reaction_li` 被乘到 \(\xi\) 方程里，决定 Li 金属相增加/减少。
- `liion_source_expr = F/Omega_Li * xi_t` 又进入 \(\phi_l\) 方程，保证 \(\Omega_1\) 内相场变化和离子电流源联系起来。

## 6. 晶粒取向相场 \(\eta_i\)

### 6.1 强式

代码写的是多晶 Allen-Cahn：

\[
\eta_{active}\frac{\eta_i^{n+1}-\eta_i^n}{\Delta t}
-
\eta_{active}L_{gb}\frac{k_{gb}}{L^2}\hat\Delta\eta_i
+
\eta_{active}L_{gb}W_{gb}\frac{\partial f_{gb}}{\partial\eta_i}
+
\eta_{active}L_{gb}W_{gb,\xi}\xi^2\eta_i
=0
\]

其中：

\[
\eta_{active}=1-h(\xi)
\]

交叉梯度项若 `kappa_gb_cross` 非零，还包含：

\[
-2\eta_{active}L_{gb}\frac{\kappa_{gb,cross}}{L^2}
\sum_{j\ne i}\hat\Delta\eta_j
\]

### 6.2 代码对应

```python
eta_grad_coeff = p.eta_mobility_scale * p.L_gb * p.k_gb / (p.length_scale**2)
eta_chem_coeff = p.eta_mobility_scale * p.L_gb * p.W_gb
eta_active = 1.0 - hxi
```

```python
F_eta += (
    eta_active * (eta_i - eta_i_n) / dt_const * v_eta_i * dx(OMEGA1)
    + eta_grad_coeff * eta_active * ufl.dot(ufl.grad(eta_i), ufl.grad(v_eta_i)) * dx(OMEGA1)
    - p.eta_mobility_scale
    * p.L_gb
    * p.kappa_gb_cross
    / (p.length_scale**2)
    * 2.0
    * eta_active
    * sum(
        ufl.dot(ufl.grad(etas[j]), ufl.grad(v_eta_i))
        for j in range(n_grains)
        if j != i
    )
    * dx(OMEGA1)
    + eta_chem_coeff * eta_active * dF_deta_i * v_eta_i * dx(OMEGA1)
    + p.eta_mobility_scale
    * p.L_gb
    * p.W_gb_xi
    * eta_active
    * xi**2
    * eta_i
    * v_eta_i
    * dx(OMEGA1)
)
```

其中：

```python
dF_deta_i = (
    2.0 * eta_i * (1.0 - eta_i) * (1.0 - 2.0 * eta_i)
    + 6.0 * eta_i * cross
)
```

检查点：

- \(\eta_i\) 只应在 \(\Omega_1\) 内有物理意义。
- 强 DOF 版本中，\(\Omega_2/\Omega_3\) 内部的 \(\eta_i\) DOF 被强制为 0。
- `eta_active = 1-h(xi)` 表示 Li 金属相内部晶粒演化被冻结/削弱。

## 7. 力学平衡方程

### 7.1 强式

各区域：

\[
\nabla\cdot\sigma=0
\]

线弹性应力：

\[
\sigma=2\mu\epsilon_e+\lambda\operatorname{tr}(\epsilon_e)I
\]

\[
\epsilon_e=\epsilon(u)-\epsilon^*
\]

本代码中：

\[
\epsilon_1^*=\beta_{Li}h(\xi)I
\]

\[
\epsilon_2^*=0
\]

\[
\epsilon_3^*=
\frac{\Omega_c c_{max}(c-c_0)}{3}I
\]

### 7.2 弱式

乘以位移测试函数 \(v_u\)，分部积分并采用自然力边界：

\[
\int_{\Omega_1}\sigma_1:\epsilon(v_u)\,d\Omega
+
\int_{\Omega_2}\sigma_2:\epsilon(v_u)\,d\Omega
+
\int_{\Omega_3}\sigma_3:\epsilon(v_u)\,d\Omega
=0
\]

代码额外乘了一个残差缩放：

\[
\texttt{mechanics\_residual\_scale}=10^{-10}
\]

这不改变数学解，但改变非线性残差不同物理方程之间的数值权重。

### 7.3 代码对应

```python
eps_u = strain(u_vec)
eig1 = p.beta_li * hxi * I2
eig2 = 0.0 * I2
eig3 = (p.omega_cathode * p.c_li_max * (c - p.soc_init) / 3.0) * I2
eps_eff1 = eps_u - eig1
eps_eff2 = eps_u - eig2
eps_eff3 = eps_u - eig3
sigma1 = 2.0 * mu1 * eps_eff1 + lam1 * ufl.tr(eps_eff1) * I2
sigma2 = 2.0 * mu2 * eps_eff2 + lam2 * ufl.tr(eps_eff2) * I2
sigma3 = 2.0 * mu3 * eps_eff3 + lam3 * ufl.tr(eps_eff3) * I2
```

```python
F_mech = p.mechanics_residual_scale * (
    ufl.inner(sigma1, strain(v_u)) * dx(OMEGA1)
    + ufl.inner(sigma2, strain(v_u)) * dx(OMEGA2)
    + ufl.inner(sigma3, strain(v_u)) * dx(OMEGA3)
)
```

检查点：

- `dfmech_dxi` 用于反馈到 \(\xi\) 方程。
- `hydro1 * omega_li / F` 用于负极机械过电势。
- `hydro3 * omega_cathode / F` 用于正极机械过电势和浓度应力迁移项。

## 8. 机械-电化学耦合项

### 8.1 负极过电势

\[
\eta_a=\phi_l-E_a^\Theta-\frac{\Omega_{Li}\sigma_h}{F}
\]

代码：

```python
hydro1 = plane_strain_hydrostatic_stress(lam1, mu1, eps_u, eig1)
overp_mech_volts = hydro1 * p.omega_li / p.F
overp_li_volts = phil - p.e_eq_li - overp_mech_volts
```

### 8.2 正极过电势

\[
\eta_c=\phi_s-\phi_l-E_{eq}(c)-\frac{\Omega_c\sigma_h}{F}
\]

代码：

```python
hydro3 = plane_strain_hydrostatic_stress(lam3, mu3, eps_u, eig3)
overp_mech_c_volts = hydro3 * p.omega_cathode / p.F
overp_c_volts = phis - phil - e_eq - e_eq_eff
```

### 8.3 浓度应力迁移项

\[
\mathbf j_c=-D\nabla c+\frac{Dc\Omega_c}{RT}\nabla\sigma_h
\]

代码体项：

```python
- (p.D_li * p.omega_cathode / (p.R * p.T * p.length_scale**2))
* c
* ufl.dot(ufl.grad(hydro3), ufl.grad(v_c))
* dx(OMEGA3)
```

检查点：

- 这里负号来自弱式中的
  \[
  -\int \frac{Dc\Omega_c}{RT}\nabla\sigma_h\cdot\nabla v_c
  \]
- `hydro3` 单位是 Pa，\(\Omega_c\sigma_h/F\) 单位是 V。

## 9. 强 DOF 约束取代 inactive penalty

原来的做法是给非物理区域加小惩罚项：

\[
\epsilon\int_{\Omega_{inactive}}u_{inactive}v\,d\Omega
\]

这个会保留非物理 DOF，并引入很小的 Jacobian 对角块，容易造成病态线性系统。

新文件中去掉了 `F_inactive`：

```python
F_total = F_xi + F_phil + F_phis + F_c + F_eta + F_mech
```

然后使用强 Dirichlet：

```python
inactive_scalar_dofs = {
    "xi_eta": inactive_only_dofs_from_markers(
        V_scalar, domain_markers, (OMEGA2, OMEGA3), (OMEGA1,)
    ),
    "phil": inactive_only_dofs_from_markers(
        V_scalar, domain_markers, (OMEGA3,), (OMEGA1, OMEGA2)
    ),
    "phis": inactive_only_dofs_from_markers(
        V_scalar, domain_markers, (OMEGA1, OMEGA3), (OMEGA2,)
    ),
    "c": inactive_only_dofs_from_markers(
        V_scalar, domain_markers, (OMEGA1, OMEGA2), (OMEGA3,)
    ),
}
```

含义：

- \(\xi,\eta_i\)：只在 \(\Omega_1\) 活跃，纯 \(\Omega_2/\Omega_3\) 内部 DOF 固定为 0。
- \(\phi_l\)：只在 \(\Omega_1+\Omega_2\) 活跃，纯 \(\Omega_3\) 内部 DOF 固定为 0。
- \(\phi_s\)：只在 \(\Omega_2\) 活跃，纯 \(\Omega_1/\Omega_3\) 内部 DOF 固定为 0。
- \(c\)：只在 \(\Omega_3\) 活跃，纯 \(\Omega_1/\Omega_2\) 内部 DOF 固定为 0。

检查点：

- 这仍然不是真正的 restricted space，因为界面共享 DOF 还存在于全局混合空间中。
- 但它比 `inactive_penalty` 稳定，因为非物理内部 DOF 不再作为软惩罚未知量参与线性系统。
- 界面共享 DOF 被保留自由，是为了不破坏 \(\Gamma_s\) 上的 `dS(GAMMA_S)` 耦合。

## 10. 守恒诊断对应关系

代码计算：

```python
gamma_s_current_A_m = assemble_total(
    i_cathode_s * p.length_scale * dS(GAMMA_S)
)
gamma_c_current_A_m = assemble_total(
    i_app * p.length_scale * ds(GAMMA_C)
)
omega1_xi_current_A_m = assemble_total(
    (p.F / p.omega_li) * xit_expr * p.length_scale**2 * dx(OMEGA1)
)
```

理论检查：

\[
G_s=\int_{\Gamma_s}i_s\,d\Gamma
\]

\[
G_c=\int_{\Gamma_c}i_{app}\,d\Gamma
\]

\[
\Xi=\int_{\Omega_1}\frac{F}{\Omega_{Li}}\frac{\xi^{n+1}-\xi^n}{\Delta t}\,d\Omega
\]

当前符号约定下应检查：

\[
G_s+G_c\approx 0
\]

\[
\Xi+G_s\approx 0
\]

也就是：

```python
gamma_s_plus_gamma_c_A_m = gamma_s_current_A_m + gamma_c_current_A_m
xi_plus_gamma_s_A_m = omega1_xi_current_A_m + gamma_s_current_A_m
```

充电时若 `i_app>0`，通常应该看到：

\[
G_c>0,\qquad G_s<0,\qquad \Xi>0
\]

这表示正极失锂、负极 Li 金属增加。

## 11. 总残差结构

代码最终：

```python
F_total = F_xi + F_phil + F_phis + F_c + F_eta + F_mech
J = ufl.derivative(F_total, w, dw)
```

对应数学上全耦合求解：

\[
F(\xi,\phi_l,\phi_s,c,u,\eta_1,\dots,\eta_N)=0
\]

其中耦合关系为：

- \(\xi\) 影响 \(\sigma_{eff}\)、力学本征应变、负极反应权重。
- \(\phi_l\) 影响负极 BV 和正极 BV。
- \(\phi_s\) 影响正极 BV。
- \(c\) 影响正极 \(E_{eq}\)、\(i_{0,c}\)、正极本征应变、应力迁移。
- \(u\) 通过 \(\sigma_h\) 影响两个过电势和 \(c\) 通量。
- \(\eta_i\) 影响晶界指标、\(\sigma_{eff}\)、\(\xi\) 的晶界惩罚。

## 12. 最需要重点检查的地方

1. \(\Gamma_s\) 上 `avg(...)` 是否与论文的指定侧 trace 一致。
   现在代码能工作，但严格来说 `phis` 应取 \(\Omega_2\) 侧，`c` 应取 \(\Omega_3\) 侧。

2. \(\phi_l\) 方程的体源符号。
   当前代码：
   \[
   -LJ_{cell}v_l
   \]
   对应强式
   \[
   \nabla\cdot(-\sigma\nabla\phi_l)=J_{cell}
   \]

3. \(c\) 方程界面项符号。
   当前代码：
   \[
   -\frac{i_s}{F c_{max}L}v_c
   \]
   来自论文 \(n=n_{\Omega_2}\) 而 \(\Omega_3\) 外法向相反。

4. 负极 `reaction_li` 与正极 `i_cathode_s` 不要混用。
   `reaction_li` 控制 \(\xi\)，`i_cathode_s` 控制 \(\Gamma_s\) 上的 \(\phi_l,\phi_s,c\) 耦合。

5. 力学残差缩放 `mechanics_residual_scale=1e-10` 不改变连续方程，但会影响牛顿残差平衡和收敛行为。

