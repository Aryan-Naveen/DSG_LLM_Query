import spark_dsg as dsg
import re
import json


def get_object_counts_per_room(G):
    # This should be a mapping between the node ID of each room and a dictionary
    # containing the category name and number of object instances for that category
    room_object_counts = {}
    
    key = G.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = G.get_labelspace(key.layer, key.partition)
    
    for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
        room_object_counts[room.id] = {}
        for name in labelspace.names_to_labels:
            room_object_counts[room.id][name] = 0
        
        for place_id in room.children():
            place = G.get_node(place_id)
            
            for object_id in place.children():
                if dsg.NodeSymbol(object_id).category != 'O':
                    continue
                object_node = G.get_node(object_id)
                room_object_counts[room.id][labelspace.labels_to_names[object_node.attributes.semantic_label]] += 1
    
    
    
    return room_object_counts


def get_objects_in_room(G, room):
    room_objects = []
    
    key = G.get_layer_key(dsg.DsgLayers.OBJECTS)
            
    for place_id in room.children():
        place = G.get_node(place_id)
        
        for object_id in place.children():
            if dsg.NodeSymbol(object_id).category != 'O':
                continue
            object_node = G.get_node(object_id)
            room_objects.append(object_node)
    
    
    
    return room_objects


def get_nested_attr(obj, attr_path, default='N/A'):
    try:
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default
    


def sanitize_world_R(s):
    # Match the key and quaternion fields
    match = re.match(r"(\w+)\s*=\s*Quaternion<(.+)>", s.strip())
    if not match:
        raise ValueError("Input does not match expected Quaternion format")

    key = match.group(1)
    components_str = match.group(2)

    # Parse the components into a dict
    components = {}
    for comp in components_str.split(','):
        k, v = comp.strip().split('=')
        components[k] = float(v)

    # Build JSON-compatible dict
    return {key: components}

def sanitize_bbox(pseudo_dict_str):    
    # Add quotes around top-level key before =
    pseudo_dict_str = re.sub(r"^(\w+)\s*=", r'"\1":', pseudo_dict_str)

    # Add quotes around all unquoted dict keys inside the braces
    # This will match any key that is a word and followed by colon
    pseudo_dict_str = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', pseudo_dict_str)

    # Surround with braces if not already
    if not pseudo_dict_str.strip().startswith("{"):
        pseudo_dict_str = "{" + pseudo_dict_str + "}"

    return json.loads(pseudo_dict_str)

def sanitize_position(s):
    # Add quotes around key
    s = re.sub(r"^(\w+)\s*=", r'"\1":', s)

    # Replace space-separated numbers inside brackets with comma-separated
    # Match the array inside []
    def replace_spaces(match):
        # Get inner content
        content = match.group(1)
        # Split on whitespace and join with comma
        nums = content.strip().split()
        return "[" + ", ".join(nums) + "]"

    s = re.sub(r"\[(.*?)\]", replace_spaces, s)

    # Wrap in braces if needed
    if not s.strip().startswith("{"):
        s = "{" + s + "}"

    # Now parse JSON
    return json.loads(s)

sanitization_function = {
    "bounding_box": sanitize_bbox,
    "position": sanitize_position,
    'world_R_object': sanitize_world_R,
}
