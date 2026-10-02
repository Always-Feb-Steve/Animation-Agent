import os
import yaml
import torch


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert device string to torch.device
    if 'device' in config:
        config['device'] = torch.device(config['device'])
    
    # Process PI fractions in the config (e.g., "PI/10")
    _process_pi_values(config)
    
    return config


def _process_pi_values(config_dict):
    """
    Process string values like "PI/10" in the config and convert them to float.
    
    Args:
        config_dict: Configuration dictionary (modified in-place)
    """
    import math
    
    def _parse_value(value):
        if isinstance(value, str) and 'PI' in value:
            try:
                # Replace PI with math.pi and evaluate
                expr = value.replace('PI', str(math.pi))
                return eval(expr)
            except:
                return value
        return value
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            _process_pi_values(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _process_pi_values(item)
                else:
                    value[i] = _parse_value(item)
        else:
            config_dict[key] = _parse_value(value)


def save_config(config, output_path):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dict
        output_path: Path to save the YAML file
    """
    # Convert torch.device to string
    if 'device' in config and isinstance(config['device'], torch.device):
        config = config.copy()  # Make a copy to avoid modifying the original
        config['device'] = str(config['device'])
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(base_config, override_config):
    """
    Update base configuration with values from override configuration.
    
    Args:
        base_config: Base configuration dict
        override_config: Override configuration dict
        
    Returns:
        Updated configuration dict
    """
    result = base_config.copy()
    
    def _update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    return _update_dict(result, override_config) 