#!/usr/bin/env python
import numpy as np
from optim import optimize
import os, sys
import time
import datetime
import shutil

from tcl.tcl import Tcl


##TODO: do better
class Parameters :
    def __init__(self):
        #data

        self.date = "2021-01-01"
        self.date_sans_tirets=self.date.replace('-','')

        self.T = 48 #nombre de pas de temps
        self.time_step_duration=0.5 #en heures

        #bornes min / max sur la puissance de l'agreget
        self.pmin= 0 #-1000
        self.pmax= 400 #3000

        #Périodes
        #définies par un couple (pas de temps debut - inclu, pas de temps fin - non inclu)
        self.period_init=(36,38) #Période de ref initiale
        self.period_eff=(38,40) #Période de ref finale
        self.period_finale=(40,42) # Période effacement
        self.largeur_rectangles=2 #largeur des rectangles d'effacement, en pas de temps. Ex: 2 --> 1h

        ##on suppose que la taille de la période d'effacement est un multiple de la largeur des rectangles
        self.nombre_rectangles= (self.period_eff[1] - self.period_eff[0]) // self.largeur_rectangles

        #rémunération effacement pour chaque rectangle de period_eff
        self.spot_prices=np.array([30 for k in range(self.nombre_rectangles)]) #np.array([30 for k in range(self.nombre_rectangles)])

        #optim params
        self.max_iter_FrankWolfe = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        self.rho = 1
        self.fully_corrective = False
        self.depth_fully_corrective = 10 # nombre des précédentes ittérations dont nous souhaitons garder la solution à leur sous pb

        ##output params
        self.output_dir = "output_optim"

# signature = {datetime.datetime.now():%Y-%m-%d_%H-%M-%S_%f}
params=Parameters()
os.makedirs(params.output_dir, exist_ok=True)
optim_dir = os.path.join(params.output_dir, f"optim_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S_%f}")
os.mkdir(optim_dir)
# sous-répertoire pour copier les config
copy_config_dir = os.path.join(optim_dir, 'config_agents')
os.mkdir(copy_config_dir)

params.output_dir = optim_dir

# Ajout des TCL:
agents_list = []
input_tcl_file = os.path.join(os.getcwd(),'OptimisationConsomationEDF','tcl','input')
for directory in os.listdir(input_tcl_file):
    file_name = os.path.join(input_tcl_file,directory,directory+'.json')
    if os.path.isfile(file_name):
        tcl = Tcl.from_json(file_name)
        agents_list.append(tcl)


##Optimisation : Frank Wolfe + recombinaison sur liste "agents_list" complete (on rajoute le suffixe "_all" aux fichiers de sorties )
suffix = "_all"
print("\nRunning optimize for all agents")
(agent_profiles_dict, agent_costs_dict, best_iteration_per_agent) = optimize(params, agents_list, suffix)

print("\nSolution recombinée :")
print(best_iteration_per_agent)

