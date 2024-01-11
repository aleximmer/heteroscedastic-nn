import yaml


def read_yaml_config(argv):
    args_dict = dict()
    if '--config' not in argv:
        return args_dict
    ix = argv.index('--config')
    # go through arguments until the end or no 'yaml' ending present
    for i in range(ix+1, len(argv)):
        if '.yaml' not in argv[i]:
            break
        with open(argv[i]) as f:
            config = yaml.full_load(f)
        # only update intersecting keys
        args_dict.update(config)
    return args_dict


def set_defaults_with_yaml_config(parser, argv):
    config = read_yaml_config(argv)
    for action in parser._actions:
        if action.choices is not None:
            val = config.get(action.dest, action.default)
            if val is not None and val not in action.choices:
                print(action.dest, action.default, action.choices)
                raise ValueError('Invalid config argument', action.dest,
                                 config[action.dest], 'Options:', action.choices)
        action.default = config.get(action.dest, action.default)
