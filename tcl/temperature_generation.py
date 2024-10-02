import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_daily_temperature_profile():
    # Définir les températures moyennes pour chaque mois en France (en Celsius)
    avg_temps = {
        "January": (5, 0),
        "February": (6, 1),
        "March": (9, 3),
        "April": (12, 6),
        "May": (16, 10),
        "June": (19, 13),
        "July": (22, 15),
        "August": (22, 15),
        "September": (19, 12),
        "October": (14, 8),
        "November": (9, 4),
        "December": (6, 1)
    }
    list_month = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    month = np.random.choice(list_month)
    # Obtenir les températures moyennes pour le mois donné
    avg_day_temp, avg_night_temp = avg_temps[month]

    # Générer une plage de dates pour une journée avec un pas de temps de 30 minutes
    #date_range = pd.date_range(start="2023-01-01 00:00:00", end="2023-01-01 23:30:00", freq="30T")

    # Initialiser une liste vide pour stocker le profil de température
    temperature_profile = []

    # Définir la dynamique de changement de température
    morning_rise = np.linspace(avg_night_temp, avg_day_temp, num=12)  # Augmentation de la température de 6h à 12h
    afternoon_stable = np.full(12, avg_day_temp)  # Température stable de 12h à 18h
    evening_fall = np.linspace(avg_day_temp, avg_night_temp, num=12)  # Diminution de la température de 18h à 0h
    night_stable = np.full(12, avg_night_temp)  # Température stable de 0h à 6h

    # Concaténer tous les changements de température pour former un profil complet de la journée
    full_day_profile = np.concatenate((night_stable, morning_rise, afternoon_stable, evening_fall))

    # Ajouter une variation aléatoire au profil de température
    temperature_profile = full_day_profile + np.random.normal(0, 1, size=full_day_profile.shape)

    # Créer un DataFrame avec la plage de dates et le profil de température
    #df = pd.DataFrame({"DateTime": date_range, "Temperature": temperature_profile})

    return temperature_profile

