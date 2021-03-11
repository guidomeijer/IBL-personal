from vedo import *
from iblviewer import *

import os
import numpy as np
import pandas as pd
import pickle

import random
import vedo


def process_priors(controller, file_path='/home/guido/Data/Ephys/Decoding/linear-regression/prior-stimside_pseudo_kfold-interleaved_aligned-behavior_all_cells_beryl-atlas.p', randomize=None):
    """
    Process priors data and get color map and scalar values
    """
    pickles = []
    with (open(os.path.abspath(file_path), 'rb')) as openfile:
        while True:
            try:
                pickles.append(pickle.load(openfile))
            except EOFError:
                break
    df = pickles[0]
    df['r_over_chance'] = df['r_prior'] - df['r_prior_null']
    filtered_df = df.groupby('region').median()['r_over_chance']

    min_value = float(np.amin(filtered_df, axis=0))
    max_value = float(np.amax(filtered_df, axis=0))
    print('Min prior value ' + str(min_value))
    print('Max prior value ' + str(max_value))

    scalars_map = {}
    for acronym, value in filtered_df.iteritems():
        if controller.model.get_region_and_row_id(acronym) is None:
            continue
        region_id, row_id = controller.model.get_region_and_row_id(acronym)
        if row_id == 0:
            # We ignore void acronym
            continue
        scalars_map[int(row_id)] = float(value)

    return scalars_map


def get_color_map(controller, scalar_map, nan_color=[0.0, 0.0, 0.0], nan_alpha=1.0):
    """
    Get a color map
    :param controller: IBLViewer.AtlasController
    :param scalar_map: Dictionary that maps scalar values in the dictionary to your custom values
    :param nan_color: Default color for unspecified values
    :param nan_alpha: Default alpha for unspecified values
    :param seed: Random seed to fake a time series
    :return: Color map and alpha map
    """
    rgb = []
    alpha = []

    # Init all to clear gray (90% white)
    #c = np.ones((self.metadata.id.size, 4)).astype(np.float32) * 0.9
    #c[:, -1] = 0.0 if only_custom_data else alpha_factor
    #print('Assigning', values.size, 'to atlas ids', self.metadata.id.size)

    for r_id in range(len(controller.model.metadata)):
        rgb.append([r_id, nan_color])
        a = nan_alpha if r_id > 0 else 0.0
        alpha.append([r_id, a])

    values = scalar_map.values()
    min_p = min(values)
    max_p = max(values)
    rng_p = max_p - min_p

    """
    # Another way to compute colors with matplotlib cm. You will need to import matplotlib
    norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    """

    for row_id in scalar_map:
        value = scalar_map[row_id]

        """
        # Another way to compute colors with matplotlib cm
        # (yields the same result as vedo, except you get alpha on top)
        r, g, b, a = mapper.to_rgba(value)
        rgb[row_id] = [row_id, [r, g, b]]
        """
        rgb[row_id] = [row_id, list(vedo.colorMap(value, 'viridis', min_p, max_p))]
        alpha[row_id] = [row_id, 1.0]

    return rgb, alpha


def load_priors_in_viewer(controller, nan_color=[0.0, 0.0, 0.0], nan_alpha=1.0):
    """
    Load priors into the viewer, faking a time series from there
    :param controller: IBLViewer.AtlasController
    :param nan_color: Default color for unspecified values
    :param nan_alpha: Default alpha for unspecified values
    """
    scalar_map = process_priors(controller)
    rgb, alpha = get_color_map(controller, scalar_map, nan_color, nan_alpha)
    controller.add_transfer_function(scalar_map, rgb, alpha, make_current=True)


if __name__ == '__main__':

    resolution = 25 # units = um
    mapping = 'Beryl'
    controller = atlas_controller.AtlasController()
    controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)

    load_priors_in_viewer(controller, nan_color=[0.75, 0.75, 0.75], nan_alpha=0.15)
    controller.render()

