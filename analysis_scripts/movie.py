# Python script to color models automatically in ChimeraX

from chimerax.core.commands import run
from chimerax.core.models import Surface
#from chimerax.core.models import list_models

# List of colors
colors = ["#6a5acd", "#98fb98", "#ff7f50", "#afeeee", "#f5deb3", "#ddaodd", "#5f9ea0", "#bc8f8f", '#9370db', '#778899',
          '#d7837f', '#b5b35c', '#b0e0e6', '#f0e68c', '#ccccff', '#d0f0c0', '#d2b48c', '#87ceeb', '#cd5c5c', '#7fffd4']

import sys, os
import numpy as np
def get_contour_level(session, vol, quantile, num_std_dev=4):
    # Fetch the volume
    # Calculate mean and standard deviation
    mean = np.mean(vol.data.full_matrix())
    std_dev = np.std(vol.data.full_matrix())
    median = np.quantile(vol.data.full_matrix(), quantile)

    # Set the contour level
    contour_level = mean + num_std_dev * std_dev
    return median
    #print(f'setting contour_level for {vol.id_string} to {contour_level}')
    #run(session, f'volume #{vol.id_string} level {contour_level}')

def main():
    print(f"{os.path.dirname(os.path.abspath(__file__))}")
    print(f"{os.getcwd()}")
    # Get the list of all models
    run(session, f'open {os.getcwd()}/{sys.argv[1]}*.mrc vseries true')
    run(session, f'lighting soft')
    run(session, f'color #1 {colors[10]}')
    #print(dir(session.models[0].volume_model(0)))
    contour = get_contour_level(session, session.models[0].volume_model(0), float(sys.argv[2]))
    run(session, f'volume #* level {contour}')
    run(session, f'vseries play #1 dir forward nor True maxFr 3')

main()
