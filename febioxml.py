axis_to_febio = {'x1': 'x',
                 'x2': 'y',
                 'x3': 'z',
                 'pressure': 'p'}

control_tagnames_to_febio = {'time steps': 'time_steps',
                             'step size': 'step_size',
                             'max refs': 'max_refs',
                             'max ups': 'max_ups',
                             'dtol': 'dtol',
                             'etol': 'etol',
                             'rtol': 'rtol',
                             'lstol': 'lstol',
                             'plot level': 'plot_level',
                             'time stepper': 'time_stepper',
                             'max retries': 'max_retries',
                             'dtmax': 'dtmax',
                             'dtmin': 'dtmin',
                             'opt iter': 'opt_iter'}
control_tagnames_from_febio = {v: k for k, v in control_tagnames_to_febio.items()}
