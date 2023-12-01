import json

import taichi as ti

from cube_object import CubeObject
from materials import Material
from preset import Preset


class ConfigLoader:

    def load_presets(self, path):
        with open(path) as j:
            cfg = json.load(j)['presets']
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

    def load_materials(self, path):
        with open(path) as j:
            cfg = json.load(j)['objects']
        materials = []
        for i in range(0, len(cfg)):
            materials.append(Material(cfg[i]))
        return materials

    def __init__(self, path):
        self.materials = self.load_materials(path)
        self.presets = self.load_presets(path)
