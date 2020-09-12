import numpy as np
import matplotlib.pyplot as plt

def vander_pol_oscillator(m, x, y):
    u, v = m*(x - x**3/3 - y), x/m
    return u, v


l = 8.0
n = 101
mu = 0.5

x1 = np.linspace(-l, l, n)
x2 = np.linspace(-l, l, n)

X1, X2 = np.meshgrid(x1, x2)

U, V = vander_pol_oscillator(mu, X1, X2)
vels = np.hypot(U, V)

slice_interval = 4
skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))

plt.figure()
plt.quiver(X1[skip], X2[skip], U[skip], V[skip])
plt.show()
