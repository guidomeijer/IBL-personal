#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:22:22 2020

@author: guido
"""

import pandas as pd
from ephys_functions import combine_layers_cortex
from ibllib.atlas import AllenAtlas
from oneibl.one import ONE
one = ONE()

# Get all brain region acronyms
ba = AllenAtlas(25)
all_regions = ba.regions.acronym

# Exclude root
incl_regions = [i for i, j in enumerate(all_regions) if not j.islower()]
all_regions = all_regions[incl_regions]

# Combine cortex layers
all_regions = combine_layers_cortex(all_regions, delete_duplicates=True)

# Get number of recordings per region
num_rec_regions = pd.DataFrame(index=all_regions, columns=['num_recordings'])
for i, region in enumerate(all_regions):
    print('Querying %d of %d' % (i+1, len(all_regions)))
    ses = one.alyx.rest('sessions', 'list', atlas_acronym=region,
                        task_protocol='_iblrig_tasks_ephysChoiceWorld',
                        project='ibl_neuropixel_brainwide')
    num_rec_regions.loc[region, 'num_recordings'] = len(ses)