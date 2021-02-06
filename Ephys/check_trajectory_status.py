
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

traj_id = 'f06d6cd9-a6b8-49a4-90d1-7905d04c2f8b'

traj = one.alyx.rest('trajectories', 'list', probe_insertion=traj_id)
eid = traj[0]['session']['id']
trials_all = one.load_object(eid, 'trials')
trials = dict()
trials['temp_key'] = trials_all
perf_easy, n_trials, _, _, _ = training.compute_bias_info(trials, trials_all)
print('Performance: %.1f%%, Number of trials: %d' % (perf_easy * 100, n_trials))
fail_behav = one.alyx.rest('trajectories', 'list', provenance='Planned', probe_insertion=traj_id,
                           django='probe_insertion__session__extended_qc__behavior,0')
if len(fail_behav) > 0:
    print("Behavior criterion FAILED")
else:
    print('Behavior criterion PASSED')
fail_crit = one.alyx.rest('trajectories', 'list', provenance='Planned',
                          django='probe_insertion__session__qc,50', id=traj_id)
if len(fail_crit) > 0:
    print("Recording set to CRITICAL")
fail_hist = one.alyx.rest('trajectories', 'list', provenance='Planned', id=traj_id,
                          django='probe_insertion__json__extended_qc__tracing_exists,False')
if len(fail_hist) > 0:
    print("Tracing does not exist")
task_qc = ['stimOn_goCue_delays', 'response_feedback_delays', 'response_stimFreeze_delays',
           'wheel_move_before_feedback', 'wheel_freeze_during_quiescence',
           'error_trial_event_sequence', 'correct_trial_event_sequence', 'n_trial_events',
           'reward_volumes', 'reward_volume_set', 'stimulus_move_before_goCue',
           'audio_pre_trial']
for i, qc_metric in enumerate(task_qc):
    fail_task_qc = one.alyx.rest(
                    'trajectories', 'list', provenance='Planned', id=traj_id,
                    django='probe_insertion__session__extended_qc___task_%s__lt,0.9' % qc_metric)
    if len(fail_task_qc) > 0:
        print('Task QC failed: %s' % qc_metric)
