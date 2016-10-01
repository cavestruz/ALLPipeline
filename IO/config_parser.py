import ConfigParser

def parse_configfile(cfgfile) :
    '''
    |
    |   Return a nested dictionary for parameter grid and pipeline
    |
    '''

    cfg_dict = {}
    Config = ConfigParser.ConfigParser()
    Config.optionxform = str # Otherwise, options are lowercased
    Config.read(cfgfile)
    for section in Config.sections() : 
        cfg_dict[section] = { option: Config.get(section, option) \
                                  for option in Config.options(section) }

    return cfg_dict        
