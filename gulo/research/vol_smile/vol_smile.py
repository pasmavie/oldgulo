import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate

S = 15000
K = 15000
σ = 0.2  
τ = 0.5  # 6months,  half of an year


def f_v(σ, τ):
    return σ * np.sqrt(τ)


def d1(S, K, v):
    return 1 / v * np.log(S / K) + v / 2


def d2(S, K, v):
    return 1 / v * np.log(S / K) - v / 2


def N(z):
    return (
        1
        / np.sqrt(2 * np.pi)
        * integrate.quad(lambda y: np.exp(-(1 / 2) * (y ** 2)), -np.inf, z)[0]
    )


def C(S, K, v):
    return S * N(d1(S, K, v)) - K * N(d2(S, K, v))


def vega(S, K, σ, τ):
    v = f_v(σ, τ)
    return (S * np.sqrt(τ) / np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (d1(S, K, v) ** 2))


def kappa(S, K, σ, τ):
    v = f_v(σ, τ)
    return (S * np.sqrt(τ) / (2 * σ * np.sqrt(2 * np.pi))) * np.exp(
        -1 / 2 * (d1(S, K, v) ** 2)
    )


v = f_v(σ, τ)



τ = 3/12  # 3 months
σ = 0.15
strikes = (80,90,100,110,120)
xs = np.arange(60,140,0.01)

for K in strikes:
    ys = []
    for S in xs:
        ys.append(kappa(S, K, σ, τ))
    sns.lineplot(xs, ys, legend="brief", label=f"K={K}")

ys = []
weights = [1/K**2 for K in strikes]
weights /= np.sum(weights)
for S in xs:
    ys.append(np.sum(weights*[vega(S, K, σ, τ) for K in strikes]))
sns.lineplot(xs, ys, legend="brief", label="avg vega")
