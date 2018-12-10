# Process the raw_map.yaml
import os
import yaml

MAP_DIR = os.path.dirname(os.path.realpath(__file__))
LS_MAP_NAME = 'ls_map.yaml'
LS_MAP_PATH = os.path.join(MAP_DIR, LS_MAP_NAME)

def create_list_mapping_from_raw():
    """
        Creates a list of mappings from the raw data to the 
    """
    raw_map_path = os.path.join(MAP_DIR, 'raw_map.yaml')
    with open(raw_map_path, 'r') as raw_map_file:
        raw_map = yaml.load(raw_map_file)

    for key in raw_map:
        # replace a single value to a list of values
        val = raw_map[key];
        raw_map[key] = [val];

    ls_map_path = os.path.join(MAP_DIR, 'ls_map.yaml');

    with open(ls_map_path, 'w') as ls_map_file:
        yaml.dump(raw_map, ls_map_file, default_flow_style=False)

def process_ls_mapping(ls_map_path):
    with open(ls_map_path, 'r') as ls_map_file:
        ls_map = yaml.load(ls_map_file)
    
    phn_index_map = {}
    for i, key in enumerate(ls_map):
        ls = ls_map[key]
        for phn in ls:
            phn_index_map[phn] = i

    phn_index_map_path = os.path.join(MAP_DIR, 'phn_index_map.yaml')
    with open(phn_index_map_path, 'w') as phn_file:
        yaml.dump(phn_index_map, phn_file, default_flow_style=False)



def main():
    #create_list_mapping_from_raw()
    process_ls_mapping(LS_MAP_PATH)

if __name__ == '__main__':
    main()