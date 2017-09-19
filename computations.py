"""Analytical computations."""

import sympy as sp

rh, rl = sp.symbols("r_h r_l")
δ, ρ, γ = sp.symbols("delta rho gamma")
vd, vp = sp.symbols("v_d v_p")

# value v of state s when action a: vsa
vph = δ*γ*vd + (1-δ) * ((1-γ)*rh + γ*vp)
vpl = (1-γ)*rl + γ*vp
vdh = γ*vd
vdl = (1-ρ)*γ*vd + ρ*γ*vp

# === Policy 1: risky policy ===
p1cond1 = sp.simplify(vdl >= vdh)  # policy n condition m
p1cond2 = sp.simplify(vph >= vpl)

vd_p1 = sp.solve(sp.Eq(vd, vdl), vd)[0]
vp_p1 = sp.solve(sp.Eq(vp, vph).subs(vd, vd_p1), vp)[0].simplify()
vd_p1 = vd_p1.subs(vp, vp_p1).simplify()

# === Policy 2: cautious policy ===
p2cond1 = sp.simplify(vdl >= vdh)
p2cond2 = sp.simplify(vpl >= vph)

vp_p2 = sp.solve(sp.Eq(vp, vpl), vp)[0]
vd_p2 = sp.solve(sp.Eq(vd, vdl).subs(vp, vp_p2), vd)[0]


# === Critical Optimalitiy Hyperplane ===
#  - where risky and cautious policy are both optimal
vp_hcrit = sp.solve(sp.Eq(vp, vpl), vp)[0]
Crit_Opt = sp.Eq(vp_hcrit, vp_p1)


# === Introducing the minimum acceptable reward value ===
rmin = sp.symbols("r_min")
Crit_vp_rmin_p1 = sp.Eq(vp_p1, rmin)
Crit_vp_rmin_p2 = sp.Eq(vp_p2, rmin)
Crit_vd_rmin_p1 = sp.Eq(vd_p1, rmin)
Crit_vd_rmin_p2 = sp.Eq(vd_p2, rmin)


# ------------------------------------------------------------------------------
#   lambdifications
# ------------------------------------------------------------------------------
lamb_vdp1 = sp.lambdify((δ, ρ, γ, rh, rl), vd_p1)
lamb_vpp1 = sp.lambdify((δ, ρ, γ, rh, rl), vp_p1)
lamb_p1cond1 = sp.lambdify((δ, ρ, γ, rh, rl, vp, vd), p1cond1)
lamb_p1cond2 = sp.lambdify((δ, ρ, γ, rh, rl, vp, vd), p1cond2)

lamb_vdp2 = sp.lambdify((δ, ρ, γ, rh, rl), vd_p2)
lamb_vpp2 = sp.lambdify((δ, ρ, γ, rh, rl), vp_p2)
lamb_p2cond1 = sp.lambdify((δ, ρ, γ, rh, rl, vp, vd), p2cond1)
lamb_p2cond2 = sp.lambdify((δ, ρ, γ, rh, rl, vp, vd), p2cond2)


if __name__ == '__main__':
    sp.init_session()

    sp.plot(sp.solve(Crit_vd_rmin_p1, γ)[0].subs({rh: 1.0, rl: 0.5,
                                                  ρ: 0.1, rmin: 0.2}),
                                                  (δ, 0, 1), ylim=(0, 1))

    # Plot the critical optimalitiy hyperplane in δ-γ space
    sp.plot(sp.solve(Crit_Opt, γ)[0].subs({rh: 1.0, rl: 0.6, ρ: 0.1}),
            (δ, 0, 1), ylim=(0, 1))
