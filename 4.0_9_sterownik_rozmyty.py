import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation

T_in = Antecedent(np.arange(15, 36, 1), 'T_in')
T_out = Antecedent(np.arange(5, 36, 1), 'T_out')
cooling_time = Consequent(np.arange(0, 61, 1), 'cooling_time')

T_in['low'] = fuzz.trimf(T_in.universe, [15, 15, 25])
T_in['medium'] = fuzz.trimf(T_in.universe, [20, 25, 30])
T_in['high'] = fuzz.trimf(T_in.universe, [25, 30, 35])
T_in['very_high'] = fuzz.gaussmf(T_in.universe, 33, 2)

T_out['low'] = fuzz.trimf(T_out.universe, [5, 5, 20])
T_out['medium'] = fuzz.trimf(T_out.universe, [15, 20, 25])
T_out['high'] = fuzz.trimf(T_out.universe, [20, 35, 35])

cooling_time['short'] = fuzz.trimf(cooling_time.universe, [0, 0, 20])
cooling_time['medium'] = fuzz.trimf(cooling_time.universe, [15, 30, 45])
cooling_time['long'] = fuzz.trimf(cooling_time.universe, [30, 60, 60])

rule1 = Rule(T_in['high'] & T_out['low'], cooling_time['long'])
rule2 = Rule(T_in['medium'] & T_out['medium'], cooling_time['medium'])
rule3 = Rule(T_in['low'] & T_out['high'], cooling_time['short'])
rule4 = Rule(T_in['high'] & T_out['high'], cooling_time['medium'])
rule5 = Rule(T_in['very_high'] & T_out['high'], cooling_time['long'])

cooling_ctrl = ControlSystem([rule1, rule2, rule3, rule4, rule5])
cooling_sim = ControlSystemSimulation(cooling_ctrl)

cooling_sim.input['T_in'] = 34
cooling_sim.input['T_out'] = 32

cooling_sim.compute()
print(f"Czas chłodzenia: {round(cooling_sim.output['cooling_time'], 2)} minut")

plt.figure()
for label, mf in T_in.terms.items():
    plt.plot(T_in.universe, mf.mf, label=label)
plt.title("Funkcje przynależności: Temperatura wnętrza")
plt.xlabel("Temperatura wewnętrzna")
plt.ylabel("Przynależność")
plt.legend()
plt.show()

plt.figure()
for label, mf in T_out.terms.items():
    plt.plot(T_out.universe, mf.mf, label=label)
plt.title("Funkcje przynależności: Temperatura na zewnątrz")
plt.xlabel("Temperatura zewnętrzna")
plt.ylabel("Przynależność")
plt.legend()
plt.show()

plt.figure()
for label, mf in cooling_time.terms.items():
    plt.plot(cooling_time.universe, mf.mf, label=label)
plt.title("Funkcje przynależności: Czas chłodzenia")
plt.xlabel("Czas chłodzenia (minuty)")
plt.ylabel("Przynależność")
plt.legend()
plt.show()