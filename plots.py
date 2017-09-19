"""Plots."""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from computations import lamb_vdp1, lamb_vpp1, lamb_p1cond1, lamb_p1cond2,\
    lamb_vdp2, lamb_vpp2, lamb_p2cond1, lamb_p2cond2


def plot_optimal_policies(δ, ρ, γ, rh, rl, xaxis="δ", yaxis="γ", prec=100,
                          riskyoptcolor="orange", cautiousoptcolor="blue",
                          ax=None):
    """Colorcode parameterregions according to where which policy is optimal.

    Parameters
    ----------
    δ : float
        the collapse probability δ
    ρ : float
        the recovery probability ρ
    γ : float
        the discount factor
    rh : float
        the high reward
    rl : float
        the low reward
    xaxis : string
        the parameter to be plotted on the xaxis (optional, default: "δ")
    yaxis : string
        the parameter to be plotted on the yaxis (optional, default: "γ")
    prec : int 
        the number of points for linspace (optional, default: 100)
    riskyoptcolor : string
        the color for the parameter region where the risky policy is optimal
        (optional, default: orange)
    cautiousoptcolor : string
        the color for the parameter region where the cautious policy is optimal
        (optional, default: blue)
    ax : None or axis object
        the ax where to polt to (optional, default: None)
    """
    params = {"δ": δ, "ρ": ρ, "γ": γ, "rh": rh, "rl": rl}

    # Getting x and y
    x = np.linspace(0, params[xaxis], prec)
    y = np.linspace(0, params[yaxis], prec)
    X, Y = np.meshgrid(x, y)

    params[xaxis] = X
    params[yaxis] = Y

    # Obtaining values --> for prosperous state
    vpp1 = lamb_vpp1(*params.values())
    vpp2 = lamb_vpp2(*params.values())

    # preparint data to plot
    data = (vpp1 < vpp2).astype(int)

    # colormap
    colors = [riskyoptcolor, cautiousoptcolor]
    cmap = mpl.colors.ListedColormap(colors)

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)


def plot_acceptal_states(δ, ρ, γ, rh, rl, rmin, state="degraded",
                         xaxis="δ", yaxis="γ", prec=100,
                         nonacceptcolor="Pink",
                         riskyacceptcolor="Lightblue",
                         cautiousacceptcolor="Yellow",
                         bothacceptcolor="Seagreen",
                         ax=None):
    """Colorcode parameterregions.

    Parameters
    ----------
    δ : float
        the collapse probability δ
    ρ : float
        the recovery probability ρ
    γ : float
        the discount factor
    rh : float
        the high reward
    rl : float
        the low reward
    rmin : float
        the minimal acceptal reward value
    state : stringing
        either "prosperous" or "degraded"
    xaxis : string
        the parameter to be plotted on the xaxis (optional, default: "δ")
    yaxis : string
        the parameter to be plotted on the yaxis (optional, default: "γ")
    prec : int
        the number of points for linspace (optional, default: 100)
    nonacceptcolor : string
        color for region where no policy is acceptable
        (optional, default: "pink")
    riskyacceptcolor : string
        color for region where only risky policy is acceptable
        (optional, default: "lightblue")
    cautiousacceptcolor : string
        color for region where only cautious policy is acceptable
        (optional, default: "yellow")
    bothacceptcolor : string
        color for region where both policies are acceptable
        (optional, default: "seagreen")
    ax : None or axis object
        the ax where to polt to (optional, default: None)
    """
    assert state is "prosperous" or state is "degraded"
    params = {"δ": δ, "ρ": ρ, "γ": γ, "rh": rh, "rl": rl}

    # Getting x and y
    x = np.linspace(0, params[xaxis], prec)
    y = np.linspace(0, params[yaxis], prec)
    X, Y = np.meshgrid(x, y)

    params[xaxis] = X
    params[yaxis] = Y
    ones = np.ones_like(X)

    # Obtaining values
    value_functions = {"prosperous": [lamb_vpp1, lamb_vpp2],
                       "degraded": [lamb_vdp1, lamb_vdp2]}
    vp1 = value_functions[state][0](*params.values())
    vp2 = value_functions[state][1](*params.values())

    # obtaining plotting data
    # 0: no policy acceptable, 1: only safe policy acceptable,
    # 2: only risky policy acceptable, 3: both policies acceptable
    p1_accept = ((vp1 > rmin)*ones).astype(int)
    p2_accept = ((vp2 > rmin)*ones).astype(int)
    p2_accept[p2_accept != 0] += 1
    data = p1_accept + p2_accept

    # colormap
    colors = [nonacceptcolor, riskyacceptcolor,
              cautiousacceptcolor, bothacceptcolor]
    cmap = mpl.colors.ListedColormap(colors)

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=3)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)


def iplot_acceptal_states(δ, ρ, γ, rh, rl, rmin, state="degraded",
                          xaxis="δ", yaxis="γ", prec=100):
    """Interactively colorcode parameterregions.

    Parameters
    ----------
    δ : float
        the collapse probability δ
    ρ : float
        the recovery probability ρ
    γ : float
        the discount factor
    rh : float
        the high reward
    rl : float
        the low reward
    rmin : float
        the minimal acceptal reward value
    state : string
        either "prosperous" or "degraded"
    xaxis : string
        the parameter to be plotted on the xaxis (optional, default: "δ")
    yaxis : string
        the parameter to be plotted on the yaxis (optional, default: "γ")
    prec : int (optional, default: 100)
        the number of points for linspace
    """

    # colors
    ov = 0.6
    accept_underboth_color = (1.0, 1.0, 0.0)
    accept_undercautious_color = (1.0-ov, 1.0, 0.0)
    accept_underrisky_color = (1.0, 1.0-ov, 0.0)
    accept_underno_color = (1.0-ov, 1.0-ov, 0.0)

    plot_acceptal_states(δ, ρ, γ, rh, rl, rmin, state=state,
                         nonacceptcolor=accept_underno_color,
                         riskyacceptcolor=accept_underrisky_color,
                         cautiousacceptcolor=accept_undercautious_color,
                         bothacceptcolor=accept_underboth_color,
                         xaxis=xaxis, yaxis=yaxis, prec=prec, ax=None)


def plot_sustainble_policies(δ, ρ, γ, rh, rl, rmin,
                             nonsuscolor="Red", riskysuscolor="Lightblue",
                             cautioussuscolor="Lightgreen", bothsuscolor="Green",
                             xaxis="δ", yaxis="γ", prec=100, ax=None):
    """Colorcode parameterregions.

    Parameters
    ----------
    δ : float
        the collapse probability δ
    ρ : float
        the recovery probability ρ
    γ : float
        the discount factor
    rh : float
        the high reward
    rl : float
        the low reward
    rmin : float
        the minimal acceptal reward value
    xaxis : string
        the parameter to be plotted on the xaxis (optional, default: "δ")
    yaxis : string
        the parameter to be plotted on the yaxis (optional, default: "γ")
    prec : int (optional, default: 100)
        the number of points for linspace
    ax : None or axis object
        the ax where to polt to (optional, default: None)
    """
    params = {"δ": δ, "ρ": ρ, "γ": γ, "rh": rh, "rl": rl}

    # Getting x and y
    x = np.linspace(0, params[xaxis], prec)
    y = np.linspace(0, params[yaxis], prec)
    X, Y = np.meshgrid(x, y)

    params[xaxis] = X
    params[yaxis] = Y
    ones = np.ones_like(X)

    # Obtaining values
    vpp1 = lamb_vpp1(*params.values())
    vpp2 = lamb_vpp2(*params.values())
    vdp1 = lamb_vdp1(*params.values())
    vdp2 = lamb_vdp2(*params.values())

    # obtaining plotting data
    # 0: no policy sustainable, 1: only safe policy sustainable,
    # 2: only risky policy sustainable, 3: both policies sustainable
    p_p1_accept = ((vpp1 >= rmin)*ones).astype(int)
    d_p1_accept = ((vdp1 >= rmin)*ones).astype(int)
    p_p2_accept = ((vpp2 >= rmin)*ones).astype(int)
    p_p2_accept[p_p2_accept != 0] += 1

    p1_sus = p_p1_accept * d_p1_accept  # only when both states are accept
    p2_sus = p_p2_accept  # only prosperous counts for policy 2

    data = p1_sus + p2_sus

    # colormap
    colors = [nonsuscolor, riskysuscolor, cautioussuscolor, bothsuscolor]
    cmap = mpl.colors.ListedColormap(colors)

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=3)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)


def plot_SOS_policies(δ, ρ, γ, rh, rl, xaxis="δ", yaxis="γ", prec=100,
                      cautiousafecolor="darkblue",
                      ax=None):
    """Colorcode parameterregions according to where which policy is SOS.

    Parameters
    ----------
    δ : float
        the collapse probability δ
    ρ : float
        the recovery probability ρ
    γ : float
        the discount factor
    rh : float
        the high reward
    rl : float
        the low reward
    xaxis : string
        the parameter to be plotted on the xaxis (optional, default: "δ")
    yaxis : string
        the parameter to be plotted on the yaxis (optional, default: "γ")
    prec : int (optional, default: 100)
        the number of points for linspace
    cautiousafecolor : string
        color for region where only cautious policy is safe
        (optional, default: "darkblue")
    ax : None or axis object
        the ax where to polt to (optional, default: None)
    """
    params = {"δ": δ, "ρ": ρ, "γ": γ, "rh": rh, "rl": rl}

    # Getting x and y
    x = np.linspace(0, params[xaxis], prec)
    y = np.linspace(0, params[yaxis], prec)
    X, Y = np.meshgrid(x, y)

    params[xaxis] = X
    params[yaxis] = Y

    # colormap
    colors = ["yellow", cautiousafecolor]
    cmap = mpl.colors.ListedColormap(colors)

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, np.ones_like(X), cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)


def plot_policies(δ, ρ, γ, rh, rl, rmin,
                  xaxis="δ", yaxis="γ", prec=100):
    """
    Plot classification of risky and cautious policy according to paradigms.

    see doc strings of functions:
    * plot_optimal_policies
    * plot_sustainble_policies
    * plot_SOS_policies
    """
    fig, ax = plt.subplots(1, 3, sharey='row', figsize=(6.5, 2))

    # colors
    ov = 0.6
    risky_opt_color = (1.0, ov, 0)
    cautious_opt_color = (1.0, 0, ov)

    both_sus_color = (ov, ov, 1.0)
    cautious_sus_color = (0.0, ov, 1.0)
    risky_sus_color = (ov, 0.0, 1.0)
    non_sus_color = (0.0, 0.0, 1.0)

    cautious_safe_color = (0.0, 1.0, ov)

    plot_optimal_policies(δ, ρ, γ, rh, rl, xaxis=xaxis, yaxis=yaxis,
						  prec=prec, riskyoptcolor=risky_opt_color,
                      	  cautiousoptcolor=cautious_opt_color,ax=ax[0])

    plot_sustainble_policies(δ, ρ, γ, rh, rl, rmin, xaxis=xaxis,
							 yaxis=yaxis, prec=prec,
							 bothsuscolor=both_sus_color,
							 cautioussuscolor=cautious_sus_color,
                             riskysuscolor=risky_sus_color,
                             nonsuscolor=non_sus_color,
                             ax=ax[1])

    plot_SOS_policies(δ, ρ, γ, rh, rl, xaxis=xaxis, yaxis=yaxis,
					  prec=prec, cautiousafecolor=cautious_safe_color,
					  ax=ax[2])


def _plot_PolicyCombinations(δ, ρ, γ, rh, rl, rmin, policy="risky",
                            xaxis="δ", yaxis="γ", prec=100, ax=None):
    """aux function to plot the paradigm combinations."""
    params = {"δ": δ, "ρ": ρ, "γ": γ, "rh": rh, "rl": rl}

    # Getting x and y
    x = np.linspace(0, params[xaxis], prec)
    y = np.linspace(0, params[yaxis], prec)
    X, Y = np.meshgrid(x, y)

    params[xaxis] = X
    params[yaxis] = Y
    ones = np.ones_like(X)

    # Obtaining values
    vpp1 = lamb_vpp1(*params.values())
    vpp2 = lamb_vpp2(*params.values())
    vdp1 = lamb_vdp1(*params.values())
    vdp2 = lamb_vdp2(*params.values())

    # policiy combinations: opt, sus, SOS
    # 0 = non opt, non sus, non SOS
    # 1 = non opt, non sus, SOS
    # 2 = non opt, sus, non SOS
    # 3 = non opt, sus, SOS
    # 4 = opt, non sus, non SOS
    # 5 = opt, non sus, SOS
    # 6 = opt, sus, non SOS
    # 7 = opt, sus, SOS
    p_p1_accept = ((vpp1 >= rmin)*ones).astype(int)
    d_p1_accept = ((vdp1 >= rmin)*ones).astype(int)
    p_p2_accept = ((vpp2*ones >= rmin)).astype(int)

    p1_sus = p_p1_accept * d_p1_accept  # only when both states are accept
    p2_sus = p_p2_accept  # only prosperous counts for policy 2

    p1_opt = (vpp1 > vpp2).astype(int)
    p2_opt = (vpp2 > vpp1).astype(int)

    p1_SOS = 0 * ones
    p2_SOS = ones

    if policy == "risky":
        data = 4*p1_opt + 2*p1_sus + p1_SOS
    elif policy == "safe":
        data = 4*p2_opt + 2*p2_sus + p2_SOS

    cv = 200/255.
    colors = [(0., 0., 0.), (0., cv, 0.), (0., 0., cv),
              (0., cv, cv), (cv, 0., 0.), (cv, cv, 0.),
              (cv, 0., cv), (cv, cv, cv)]
    cmap = mpl.colors.ListedColormap(colors)

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=7)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)


def plot_PolicyCombinations(δ, ρ, γ, rh, rl, rmin,
                            xaxis="δ", yaxis="γ", prec=100):
    """Plot the paradigm combinations."""
    fig, ax = plt.subplots(1, 2, sharey='row', figsize=(6.5, 3))

    _plot_PolicyCombinations(δ, ρ, γ, rh, rl, rmin, policy="risky",
                             xaxis=xaxis, yaxis=yaxis, prec=prec,
                             ax=ax[0])
    ax[0].set_title("risky policy")
    _plot_PolicyCombinations(δ, ρ, γ, rh, rl, rmin, policy="safe",
                             xaxis=xaxis, yaxis=yaxis, prec=prec,
                             ax=ax[1])
    ax[1].set_title("safe policy")

