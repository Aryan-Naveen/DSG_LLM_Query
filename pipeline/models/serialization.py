import spark_dsg as dsg
from typing import List, Dict
import json
from .utils import get_object_counts_per_room, sanitization_function, get_objects_in_room



def indented_encoding(scene_graph: dsg.DynamicSceneGraph, detail_keys: List [str]) -> str:
    encoding = ''
    key = scene_graph.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = scene_graph.get_labelspace(key.layer, key.partition)
    
    room_object_counts = get_object_counts_per_room(scene_graph)    
    for room in scene_graph.get_layer(dsg.DsgLayers.ROOMS).nodes:
        encoding += f'Room (id = {room.id.category_id}) \n'
        object_counts = room_object_counts[room.id]
        encoding += '\tRoom Object Summary:\n'
        for obj in object_counts:
            if object_counts[obj] == 0:
                continue
            encoding += (
                f'\t\t- {object_counts[obj]} {obj}s\n'
                if object_counts[obj] > 1
                else f'\t\t- {object_counts[obj]} {obj}\n'
            )

        if "NA" not in detail_keys:
            encoding += '\tRoom Object Attributes:\n'
            for place_id in room.children():
                place = scene_graph.get_node(place_id)
                
                for object_id in place.children():
                    if dsg.NodeSymbol(object_id).category != 'O':
                        continue

                    object_node = scene_graph.get_node(object_id)
                    
                    props = ", ".join(
                        f"{key}={getattr(object_node.attributes, key, 'N/A')}"
                        for key in detail_keys
                    )
                    encoding += f'\t\t- {labelspace.labels_to_names[object_node.attributes.semantic_label]} (id = {object_node.id.category_id}, {props}) \n'
    
    encoding += "Edges (Room layer):\n"
    for edge in scene_graph.get_layer(dsg.DsgLayers.ROOMS).edges:
        encoding += f'\t - Room(id={dsg.NodeSymbol(edge.source).category_id}) <----> Room(id={dsg.NodeSymbol(edge.target).category_id})\n'

    return encoding

def json_encoding(scene_graph: dsg.DynamicSceneGraph, detail_keys: List [str]) -> str:
    encoding = {"rooms": []}
    key = scene_graph.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = scene_graph.get_labelspace(key.layer, key.partition)
    
    room_object_counts = get_object_counts_per_room(scene_graph)    
    for room in scene_graph.get_layer(dsg.DsgLayers.ROOMS).nodes:
        room_encoding = {
            'id': room.id.category_id,
            'neighbor_rooms': [scene_graph.get_node(neighbor_id).id.category_id for neighbor_id in room.siblings()],
            'objects': {
                'count summary': {
                    k: v for k, v in room_object_counts[room.id].items() if v != 0
                }
            }
        }
        
        if 'NA' not in detail_keys: 
            object_details = {}
            for place_id in room.children():
                place = scene_graph.get_node(place_id)
                
                for object_id in place.children():
                    if dsg.NodeSymbol(object_id).category != 'O':
                        continue

                    object_node = scene_graph.get_node(object_id)
                    
                    object_key = labelspace.labels_to_names[object_node.attributes.semantic_label]
                    if object_key not in object_details: object_details[object_key] = {}
                    
                    object_details[object_key][f"id: {object_node.id.category_id}"] = {
                        k: v
                        for key in detail_keys
                        for k, v in sanitization_function[key](f"{key}={getattr(object_node.attributes, key, 'N/A')}").items()
                    }
                    
                    
            room_encoding["objects"]['attributes'] = object_details
            
        encoding["rooms"].append(room_encoding)

    
    return json.dumps(encoding, indent=2)


def triplets_encoding(scene_graph: dsg.DynamicSceneGraph, detail_keys: List [str]) -> str:
    encoding = ''
    key = scene_graph.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = scene_graph.get_labelspace(key.layer, key.partition)
    
    room_object_counts = get_object_counts_per_room(scene_graph)    
    for room in scene_graph.get_layer(dsg.DsgLayers.ROOMS).nodes:
        room_id = f"Room [id = {room.id.category_id}]"
        for obj in room_object_counts:
            if room_object_counts[obj] == 0:
                continue
            encoding += f" \t ({room_id}, has, {obj} [count = {room_object_counts[obj]}]) \t | "

        if 'NA' not in detail_keys:
            for place_id in room.children():
                place = scene_graph.get_node(place_id)
                
                for object_id in place.children():
                    if dsg.NodeSymbol(object_id).category != 'O':
                        continue

                    object_node = scene_graph.get_node(object_id)
                    object_category = labelspace.labels_to_names[object_node.attributes.semantic_label]

                    object_id = f"{object_category} [room_id = {room.id.category_id} object_id = {object_node.id.category_id}]"
                    props = ", ".join(
                        f"{key}={getattr(object_node.attributes, key, 'N/A')}"
                        for key in detail_keys
                    )
                    attributes_id = f"Attributes [{props}]"
                    encoding += f" \t ({object_id}, has, {attributes_id}) \t |"

    for edge in scene_graph.get_layer(dsg.DsgLayers.ROOMS).edges:
        src_room_id = f"Room [id = {dsg.NodeSymbol(edge.source).category_id}]"
        target_room_id = f"Room [id = {dsg.NodeSymbol(edge.target).category_id}]"
        encoding += f" \t ({src_room_id}, connects to, {target_room_id}) \t |"

    return encoding

def natural_lang_encoding(scene_graph: dsg.DynamicSceneGraph, detail_keys: List [str]) -> str:
    encoding = '~~~~~~~~~~ ROOM DESCRIPTIONS ~~~~~~~~~~\n'
    key = scene_graph.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = scene_graph.get_labelspace(key.layer, key.partition)
    
    room_object_counts = get_object_counts_per_room(scene_graph)  
    
    def generate_room_descriptor(room):
        
        def summarize_objects(obj_counts):
            nonzero_items = [f"{v} of {k}" for k, v in obj_counts.items() if v > 0]
            if not nonzero_items:
                return "Inside this room there are no objects."
            if len(nonzero_items) == 1:
                return f"Inside this room there is {nonzero_items[0]}."
            return f"Inside this room there are {', '.join(nonzero_items[:-1])}, and {nonzero_items[-1]}. \n"

        def describe_objects(obj_nodes, detail_keys):
            descriptions = []

            for obj_node in obj_nodes:
                # Collect and format attributes
                obj_name = labelspace.labels_to_names[obj_node.attributes.semantic_label]
                attributes = {
                    key: getattr(obj_node.attributes, key, "N/A")
                    for key in detail_keys
                }
                attr_descriptions = [f"{k} is {v}" for k, v in attributes.items() if v != "N/A"]
                attr_sentence = ", ".join(attr_descriptions[:-1])
                if attr_descriptions:
                    attr_sentence += f", and {attr_descriptions[-1]}" if len(attr_descriptions) > 1 else attr_descriptions[0]
                else:
                    attr_sentence = "No attributes available"

                paragraph = f"The {obj_name} has the following attributes: {attr_sentence}."
                descriptions.append(paragraph)

            return "\n".join(descriptions)

        intro = f'\n\nROOM {room.id.category_id} SUMMARY: \n'
        object_summary_descriptor = summarize_objects(room_object_counts[room.id])
        
        room_objects_nodes = get_objects_in_room(scene_graph, room)
        object_attribute_descriptor = ""
        if 'NA' not in detail_keys: object_attribute_descriptor = describe_objects(room_objects_nodes, detail_keys)
        
        return f"{intro}{object_summary_descriptor}\n{object_attribute_descriptor}"
    
    def generate_edge_descriptor(edge):
        src_room_id = dsg.NodeSymbol(edge.source).category_id
        target_room_id = dsg.NodeSymbol(edge.target).category_id
        return f"Room {src_room_id} is connected to {target_room_id}. "
        
    
    for room in scene_graph.get_layer(dsg.DsgLayers.ROOMS).nodes:
        encoding += generate_room_descriptor(room)

    encoding += '\n\n~~~~~~~~~~ ROOM LAYOUT SUMMARY ~~~~~~~~~~\n'
    encoding += 'Additionally several rooms are connected to each other as follows:\n'
    for edge in scene_graph.get_layer(dsg.DsgLayers.ROOMS).edges:
        encoding += generate_edge_descriptor(edge)
        
    return encoding

serialization_functions = {
    "indented": indented_encoding,
    "json": json_encoding,
    "triplets": triplets_encoding,
    "natural": natural_lang_encoding,
}


