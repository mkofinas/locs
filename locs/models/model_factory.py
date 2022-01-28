"""Factory for easily getting networks by name."""

from importlib import import_module


def acronym_case(x):
    acronyms = {'egnn': 'EGNN', 'dnri': 'DNRI', 'locs': 'LoCS',
                'dynamicvars': 'DynamicVars'}
    print(x, acronyms.get(x, x))
    return acronyms.get(x, x.capitalize())


class NetworkFactory(object):
    @staticmethod
    def create(model_name, dynamic_vars, *args, **kwargs):
        dynamic_vars_ext = '_dynamicvars' if dynamic_vars else ''
        module_name = f'{model_name}{dynamic_vars_ext}'
        class_name = ''.join([acronym_case(s) for s in module_name.split('_')])

        try:
            model_module = import_module('locs.models.' + module_name)
            model_class = getattr(model_module, class_name)
            model_instance = model_class(*args, **kwargs)
        except (AttributeError, ImportError):
            raise

        return model_instance
