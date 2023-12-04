WATER = 0
BOX = 1


class Material:
    def __init__(self, cfg):
        self.id = cfg["id"]
        self.name = cfg["name"]
        self.color = cfg["default color"]
