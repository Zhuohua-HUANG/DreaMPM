class Material:
    def __init__(self, cfg):
        self.id = cfg["id"]
        self.name = cfg["name"]
        self.color = tuple(cfg["default color"])

    def set_color(self, color):
        self.color = color

