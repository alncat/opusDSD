# Python script to color models automatically in ChimeraX

from chimerax.core.commands import run
from chimerax.core.models import Surface
#from chimerax.core.models import list_models

# List of colors
colors = ["#7f78ff", "#98fb98", "#ff7f50", "#b26e55", "#f5deb3", "#dda0dd", "#5f9ea0", "#bc8f8f", '#778899',
          '#d7837f', '#b5b35c', '#b0e0e6', '#f0e68c', '#ccccff', '#d0f0c0', '#163f22', '#87ceeb', '#cd5c5c', '#7fffd4']

import sys, os
import numpy as np

def set_contour_level(session, vol, quantile, num_std_dev=4):
    # Fetch the volume
    # Calculate mean and standard deviation
    mean = np.mean(vol.data.full_matrix())
    std_dev = np.std(vol.data.full_matrix())
    median = np.quantile(vol.data.full_matrix(), quantile)

    # Set the contour level
    contour_level = mean + num_std_dev * std_dev
    print(f'setting contour_level for {vol.id_string} to {median}, {contour_level/median}')
    run(session, f'volume #{vol.id_string} level {median}')

# Usage Example:
# Replace '1' with the ID of your map
# You can change '1.5' to the desired number of standard deviations
import re
def main():
    print(f"{os.path.dirname(os.path.abspath(__file__))}")
    print(f"{os.getcwd()}")
    # Get the list of all models
    prefix = sys.argv[1]
    pattern = re.compile(f'{re.escape(prefix)}(\d+).mrc')
    files_in_dir = os.listdir(os.path.join(os.getcwd()))
    sorted_files = sorted(files_in_dir)
    files = []
    for file in sorted_files:
        if pattern.match(file):
            files.append(file)
    for i in range(len(files)):
            run(session, f'open {os.getcwd()}/{prefix}{str(i)}.mrc')
        #run(session, f'open {os.getcwd()}/{sys.argv[1]}{i}.mrc')
    models = session.models

    # Loop through the models and assign colors
    j = 0
    for i, model in enumerate(models):
        if isinstance(model, Surface):
            continue
        print('model', model.id_string)
        color = colors[j % len(colors)]  # Cycle through the color list
        j += 1
        run(session, f'color #{model.id_string} {color}')
        set_contour_level(session, model, float(sys.argv[2]))
        #for j, jmod in enumerate(model.child_models()):
        #    color = colors[j % len(colors)]  # Cycle through the color list
        #    if isinstance(jmod, Surface):
        #        continue
        #    print('child model', jmod.id_string)
        #    run(session, f'color #{jmod.id_string} {color}')
    run(session, f'lighting soft')
main()
