import taichi as ti
import yaml

from .objects import CubeObject
from .materials import Material
from .preset import Preset


"""
Load config from config yaml file
"""
class ConfigLoader:

    def load_presets(self):
        cfg = self.config['presets']
        presets = []
        for i in range(0, len(cfg)):
            objects = []
            for j in range(0, len(cfg[i]["material"])):
                cube_object = CubeObject(ti.Vector(cfg[i]["minimum"][j]), ti.Vector(cfg[i]["size"][j]),
                                         cfg[i]["material"][j], self.materials)
                objects.append(cube_object)
            preset = Preset(cfg[i]["name"], objects)
            presets.append(preset)
        return presets

    def load_materials(self):
        cfg = self.config['objects']
        materials = []
        for i in range(0, len(cfg)):
            materials.append(Material(cfg[i]))
        return materials

    def __init__(self, path):
        self.config = yaml.safe_load(open(path))
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.G_number = self.config["grid number"]
        self.max_timestep = self.config["max timestep"]
        self.max_hard = self.config["max hard"]
        self.materials = self.load_materials()
        self.presets = self.load_presets()
