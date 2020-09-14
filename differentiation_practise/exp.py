import numpy as np
import matplotlib.pyplot as plt


def forwards(exp):
    forwards_ = (np.array(exp[1:]) - np.array(exp[:-1])).tolist()
    forwards_.append(exp[-1])
    return forwards_

fig, axs = plt.subplots(1,2)
print(axs.shape)

h_values = [0.1, 0.01]
exacts_d = []


for i,h_value in enumerate(h_values):
    range_ = np.linspace(h_value, 1-h_value, num=100)
    exp = np.exp(range_).tolist()
    exact_derivatives = exp
    exacts_d.append(exact_derivatives)
    fwd = forwards(exp)
    assert forwards([1,2,3,4,5]) = [1,1,1,1,5]

    axs[i].plot(range_, exact_derivatives)
    axs[i].set_title(str(h_value))

plt.show()


