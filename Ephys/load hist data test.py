#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:06:42 2020

@author: guido
"""

def get_subjects(self):
        """
        Finds all subjects that have a histology track trajectory registered
        :return subjects: list of subjects
        :type subjects: list of strings
        """
        sess_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
        subjects = [sess['session']['subject'] for sess in sess_with_hist]
        subjects = np.unique(subjects)

        return subjects

    def get_sessions(self, subject):
        """
        Finds all sessions for a particular subject that have a histology track trajectory
        registered
        :param subject: subject name
        :type subject: string
        :return session: list of sessions associated with subject, displayed as date + probe
        :return session: list of strings
        """
        self.subj = subject
        self.sess_with_hist = one.alyx.rest('trajectories', 'list', subject=self.subj,
                                            provenance='Histology track')
        session = [(sess['session']['start_time'][:10] + ' ' + sess['probe_name']) for sess in
                   self.sess_with_hist]
        return session