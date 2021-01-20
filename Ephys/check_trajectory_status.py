
'''
Check status of specific trajectory, if:
- failed behavior
- failed histology tracing
- fail session QC
and compute behavioral parameters
'''
import brainbox.behavior.training as training
from oneibl.one import ONE
one = ONE()
traj_id = '46656bfe-f23f-45d3-93f4-2e17d747092c'
fail_behav = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     django='probe_insertion__session__extended_qc__behavior,0', id=traj_id)
if len(fail_behav) > 0:
    print("fail_behav")
# Critical session label
fail_crit = one.alyx.rest('trajectories', 'list', provenance='Planned',
                          django='probe_insertion__session__qc,50', id=traj_id)
if len(fail_crit) > 0:
    print("fail_crit")
# Histology failing
fail_hist = one.alyx.rest('trajectories', 'list', provenance='Planned',
                          django='probe_insertion__json__extended_qc__tracing_exists,False', id=traj_id)
if len(fail_hist) > 0:
    print("fail_hist")
traj = one.alyx.rest('trajectories', 'list', id=traj_id)
eid = traj[0]['session']['id']
trials_all = one.load_object(eid, 'trials')
trials = dict()
trials['temp_key'] = trials_all
perf_easy, n_trials, _, _, _ = training.compute_bias_info(trials, trials_all)
print(f'Perf: {perf_easy}, n trials: {n_trials}')
good_enough = training.criterion_delay(n_trials, perf_easy)