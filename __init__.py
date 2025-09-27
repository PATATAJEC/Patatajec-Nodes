import os
from .nodes import *

# katalog z assetami przeglądarkowymi (JS)
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

NODE_CONFIG = {
     "PathTool": {"class": PathTool, "name": "Path Tool"},
     "ColorMatchFalloff": {"class": ColorMatchFalloff, "name": "Color Match Falloff"},
     "SequenceContentZoom": {"class": SequenceContentZoom, "name": "Sequence Content Zoom"},
     "SequenceBlend": {"class": SequenceBlend, "name": "Sequence Blend"},
     "ColorPicker": {"class": ColorPicker, "name": "Color Picker"},
     "ImageDifferenceToAplpha": {"class": ImageDifferenceToAlpha, "name": "Image Difference to Alpha"},

}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("### Loading: ComfyUI-PatatajecNodes (Success) ###")
