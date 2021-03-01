
from datoviz import canvas, run, colormap  # Needs to be the first thing to import

import numpy as np
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

eid = 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8'

spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

c = canvas(show_fps=False)

panel = c.panel(controller='arcball')
visual = panel.visual('point', depth_test=True)

clusters_pass = np.where(clusters['probe00']['metrics']['label'] == 1)[0]

amps = spikes['probe00']['amps'][np.isin(spikes['probe00']['clusters'], clusters_pass)]
spike_clusters = spikes['probe00']['clusters'][np.isin(spikes['probe00']['clusters'], clusters_pass)]
spike_depths = spikes['probe00']['depths'][np.isin(spikes['probe00']['clusters'], clusters_pass)]
spike_times = spikes['probe00']['times'][np.isin(spikes['probe00']['clusters'], clusters_pass)]

N = len(spike_times)
C = len(np.unique(spike_clusters))
print(f"{N} spikes")
print(f"{C} neurons")
pos = np.c_[spike_times, spike_depths, amps * 1000000]
color = colormap(spike_clusters.astype(np.float64), cmap='glasbey', alpha=0.5)

visual.data('pos', pos)
visual.data('color', color)
visual.data('ms', np.array([5.]))

run()