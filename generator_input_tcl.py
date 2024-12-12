import os
import json
from numpy import random
import pickle
import argparse

from tcl.temperature_generation import generate_daily_temperature_profile

def construction_pmax(coefDeltaTemp,coefConso, ecart_temperature_max,minimal_power):
    pmax = max(minimal_power, ecart_temperature_max * coefDeltaTemp / coefConso)
    return pmax

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script pour créer des TCLs.")
    
    parser.add_argument(
        "-ntcl",
        type=int,
        default=50,
        help="Le nombre de TCL à créer (par défaut : 50)."
    )
    
    args = parser.parse_args()
    
    number_of_tcl_to_create = args.ntcl
    
    for k in range(1,number_of_tcl_to_create+1):
        nom_tcl = "tcl"+str(k)
        os.makedirs(os.path.join("tcl/input",nom_tcl), exist_ok=True)
        os.makedirs(os.path.join("tcl/output",nom_tcl), exist_ok=True)

        dico = {}
        dico["name"] = nom_tcl
        dico["input_path"] = os.path.join("tcl","input",nom_tcl)
        dico["output_path"] = os.path.join("tcl","output",nom_tcl)
        dico["working_dir"] = os.path.join("tcl","input",nom_tcl)
        dico["initial_temperature"] = random.uniform(17.5,19.5)
        dico["coefDeltaTemp"] = 0.05625 # random.uniform(0.05625, 0.05625+0.015)
        dico["coefConso"] = 1. #random.uniform(1., 1.)
        dico["puissanceMin"] = 0
        pmax = construction_pmax( dico["coefDeltaTemp"],dico["coefConso"] , 30, 4)
        dico["puissanceMax"] = pmax
        dico["temperature_max"] = random.uniform(20,22)
        dico["temperature_min"] = random.uniform(15,17)
        dico["chroniqueTempExt"] = list(generate_daily_temperature_profile())

        with open('tcl/input/elec_price.pkl', 'rb') as fichier:
            elec_price = pickle.load(fichier)
        dico["electricity_cost"] =elec_price
        json_path = os.path.join("tcl/input",nom_tcl,nom_tcl+'.json')
        with open(json_path, 'w') as fichier:
            json.dump(dico, fichier, indent=4)

    print(f"{number_of_tcl_to_create} crés")