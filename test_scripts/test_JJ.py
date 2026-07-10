from classes.JJ import JJ
import qutip as qt
import numpy
import matplotlib.pyplot as plt
from classes.RC import RC
import os


training_data_size = 1000
testing_data_size = 300
alpha = 1.5
dt = 1
delay = 1
training_data = RC.generate_mackey_glass(training_data_size,dt=3,tau=50)
testing_data = RC.generate_mackey_glass(testing_data_size,dt=3,tau=50)
washout = 100


JJ_RC = JJ.JJ(window_size=0,delay=delay,training_data=training_data[washout:-delay],training_targets=training_data[washout+delay:],alpha = 1,dt = 0.01)

JJ_RC.train()

print(f"Weight matrix is: \n{JJ_RC.W}")

result = JJ_RC.test(test_data=testing_data[:-delay],test_targets=testing_data[delay:])

print(f"Error of :{result[1]}")

JJ_RC.plot(save_dir=os.path.join(os.getcwd(),"test_plots\\JJ_test"),
                    save_fig=True)



