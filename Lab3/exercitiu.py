from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# model: cutremur --influent-->incendiu; 
        # cutremur --influent-->alarma;
        # incendiu --influent-->alarma;
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])

# def tabelul de probabilitati conditionate pt fiecare variabila

cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
# C poate avea 2 valori: 0 sau 1; P(c=0)= 1-0,05% ; P(c=1) = 0,05%

cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, 
                          values=[[0.99, 0.97], [0.01, 0.03]],
                          evidence=['Cutremur'], evidence_card=[2])
# [0.99, 0.97] : P(I|C=0) = 1-0,01% = 99% ; P(I|C=1) = 1-0,03% = 97%
# [0.01, 0.03] : P(I=0|C=0) =1%; P(I=0|C=1) = 3%

cpd_alarma = TabularCPD(variable='Alarma', variable_card=2, 
                        values=[[0.9999, 0.98, 0.02, 0.95],
                                [0.0001, 0.02, 0.98, 0.05]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])
# p(alarma) in functie de cutremur si incendiu
# [0.9999, 0.98, 0.02, 0.95]: P(A|C=0 & I=0)=100%-0,01%, P(A|C=1 & I=0)-100%-2%, P(A|C=0 & I=1)=100%-0,05%, P(A|C=1 & I=1)=100%-0,02%


model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma) # adaug tabelele la model

assert model.check_model() # verif daca e ok modelul

infer = VariableElimination(model)

result = infer.query(variables=['Cutremur'], evidence={'Alarma': 1})
print('P(Cutremur|Alarma=1)', result)

result2 = infer.query(variables=['Incendiu'], evidence={'Alarma': 0})
print('P(Incendiu|Alarma=0)', result2)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()