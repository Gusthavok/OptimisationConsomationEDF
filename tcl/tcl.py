import numpy as np
import os
import json
from linopy import Model
from random import random

from aggregator.agent import Agent


class Tcl (Agent):

    @staticmethod
    def from_json(json_path: str, **args):
        data = json.load(open(json_path, "r"))
        data.update(args)
        return Tcl(**data)

    def __init__(self, name: str, input_path: str, output_path: str, working_dir: str,initial_temperature=0,
                 coefDeltaTemp = 0,coefConso = 0,puissanceMax = 0,puissanceMin = 0, temperature_max = 0,
                 temperature_min = 0, chroniqueTempExt= [0 for _ in range(48)], electricity_cost = [0 for _ in range(48)]):
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.working_dir = working_dir
        # self.consumer_directory = consumer_directory

        # Data relative to dynamics od the temperature
        self.initial_temperature = initial_temperature
        self.coefDeltaTemp = coefDeltaTemp
        self.coefConso = coefConso
        self.puissanceMax = puissanceMax
        self.puissanceMin = puissanceMin
        self.temperature_max = temperature_max
        self.temperature_min = temperature_min
        self.chroniqueTempExt = chroniqueTempExt
        # Data relative to objective function
        self.electricity_cost = electricity_cost

        self.dt = 0.5  # pas de temps de 30 minutes

    def individual_cost(self, load_profile):
        """
        Given a load profile, returns the local cost of the agent considering this profile
        """
        cost = 0
        T = len(load_profile)

        # electricity cost
        for t in range(T):
            cost += load_profile[t] * self.electricity_cost[t]*self.dt

        return cost

    def temperature_profile_simulation(self, load_profile):

        T = len(load_profile)

        temp_profile = np.zeros(T)
        temp_profile[0] = self.initial_temperature
        for t in range(1,T):
            temp_profile[t] = temp_profile[t - 1] + load_profile[t] * self.coefConso*self.dt + self.dt*self.coefDeltaTemp*(self.chroniqueTempExt[t-1]-temp_profile[t-1])
        return temp_profile

    def solve_optim_model(self, signal):
        m = Model()
        T = len(self.electricity_cost)
        p = {}
        state = {}
        obj_expr = 0
        for t in range(T):
            z = m.add_variables(name="z_{}".format(t), lower=0, upper=1, vartype="binary")

            p[t] = self.puissanceMin + z * (self.puissanceMax - self.puissanceMin)
            obj_expr += p[t] * (self.electricity_cost[t] *self.dt + signal[t])

            state[t] = m.add_variables(name="temperature_instant_{}".format(t), lower=self.temperature_min, upper=self.temperature_max)

            if t == 0:
                m.add_constraints(lhs = state[t], sign='=', rhs = self.initial_temperature, name = "initial temp" )
            else:
                m.add_constraints(lhs = state[t] ,  sign='=', rhs =  (state[t - 1] + p[t] * self.coefConso*self.dt + self.dt*self.coefDeltaTemp*self.chroniqueTempExt[t-1]-self.dt*self.coefDeltaTemp*state[t-1]) , name = f'dynamics time {t}')

        obj = m.add_objective(obj_expr)  # minimisation par défaut?
  #      fn = Path('/home/i52980/Documents/habitats_connectes/refacto_agregosiris/tcl/local.lp' )
 #       m.to_file(fn)
        m.solve()
        total_load = np.zeros(T)
        total_cost = 0
        for t in range(T):
            total_load[t] = float(p[t].solution)
            total_cost += total_load[t] * (self.electricity_cost[t]*self.dt + signal[t])


        return (total_cost, total_load)

    def tcl_update_load(self,signal, path= None):
        (total_cost, total_load) = self.solve_optim_model(signal)
        dico = {"load" : list(total_load),
                'cost' : total_cost
        }
        if path:
            json_path = os.path.join(path,self.output_path, self.name+'.json')
        else :
            json_path = os.path.join(self.output_path, self.name + '.json')
        with open(json_path, 'w') as fichier:
            json.dump(dico, fichier, indent=4)

        return None

    def read_output(self,path = None):

        if path:
            path = os.path.join(path,self.output_path,self.name+'.json')
        else :
            path = os.path.join(self.output_path, self.name + '.json')
        dico_output = json.load(open(path, "r"))
        return dico_output["load"],dico_output['cost']
    
    def get_passive_load(self):
        phase_croissante = random()>.5
        current_temperature = self.initial_temperature

        T = len(self.electricity_cost)

        total_load = np.zeros(T)
        total_cost = 0
        temperature = np.zeros(T)

        for t in range(T):
            if not phase_croissante and current_temperature<self.temperature_min:
                phase_croissante=True

            if phase_croissante and current_temperature < self.temperature_max:
                total_load[t] = self.puissanceMax
            else:
                total_load[t] = self.puissanceMin
            total_cost += total_load[t] * (self.electricity_cost[t]*self.dt + signal[t])

            temperature[t] = current_temperature
            current_temperature = current_temperature + total_load[t] * self.coefConso*self.dt + self.dt*self.coefDeltaTemp*self.chroniqueTempExt[t-1]-self.dt*self.coefDeltaTemp*current_temperature

        return total_load, total_cost, temperature






if __name__ == '__main__':


    tcl1 = Tcl.from_json("input/tcl10/tcl10.json")
    signal = [0 for _ in range(48)]
    (total_cost, total_load) = tcl1.solve_optim_model(signal)
    tcl1.tcl_update_load(signal)
    tcl1.read_output()
    print(f'total load {total_load}')
    print(f'total cost {total_cost}')
    temp_profile = tcl1.temperature_profile_simulation(total_load)
    print(f'temp profile {temp_profile}')