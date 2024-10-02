import os
import json
import sys
from numpy import random
import pickle

from tcl import Tcl
from temperature_generation import generate_daily_temperature_profile

def construction_pmax(coefDeltaTemp,coefConso, ecart_temperature_max,minimal_power):
    pmax = max(minimal_power, ecart_temperature_max * dico["coefDeltaTemp"] / dico["coefConso"])
    return pmax

if __name__ == '__main__':

    number_of_tcl_to_create = 40
    for k in range(2,number_of_tcl_to_create):
        nom_tcl = "tcl"+str(k)
        os.makedirs(os.path.join("input",nom_tcl), exist_ok=True)
        os.makedirs(os.path.join("output",nom_tcl), exist_ok=True)

        dico = {}
        dico["name"] = nom_tcl
        dico["input_path"] = os.path.join("tcl","input",nom_tcl)
        dico["output_path"] = os.path.join("tcl","output",nom_tcl)
        dico["working_dir"] = os.path.join("tcl","input",nom_tcl)
        dico["initial_temperature"] = random.uniform(17,20)
        dico[ "coefDeltaTemp"] = 0.05625
        dico["coefConso"] = 1
        dico["puissanceMin"] = 0
        pmax = construction_pmax( dico["coefDeltaTemp"],dico["coefConso"] , 30,4  )
        dico["puissanceMax"] = pmax
        dico["temperature_max"] = random.uniform(20,22)
        dico["temperature_min"] = random.uniform(15,17)
        dico["chroniqueTempExt"] = list(generate_daily_temperature_profile())

        with open('input/elec_price.pkl', 'rb') as fichier:
            elec_price = pickle.load(fichier)
        dico["electricity_cost"] =elec_price
        json_path = os.path.join("input",nom_tcl,nom_tcl+'.json')
        with open(json_path, 'w') as fichier:
            json.dump(dico, fichier, indent=4)
