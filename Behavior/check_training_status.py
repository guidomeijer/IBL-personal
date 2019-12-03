# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:39:33 2019

@author: guido
"""

from oneibl.one import ONE
import datetime
one = ONE()
eids, ses_info = one.search(users='ines',
                            date_range=[str(datetime.date.today() - datetime.timedelta(days=5)),
                                        str(datetime.date.today())],
                            details=True)