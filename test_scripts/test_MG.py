import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.generate import generate_mackey_glass
import matplotlib.pyplot as plt

x = generate_mackey_glass(2000,tau=50,dt=1)[1500:]


plt.plot(x)
plt.title("Mackey Glass Series")
plt.xlabel("Time Step")
plt.ylabel("Magnitude")
plt.show()