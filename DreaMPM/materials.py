DIY_MATERIAL = 0
SOLID_CUBE = 1

class Material:
    def __init__(self, cfg):
        self.id = cfg["id"]
        self.name = cfg["name"]
        self.color = cfg["default color"]
