PLY_NUM=48256
# PLY_NUM =140875
# PLY_NUM = 5532
# PLY_NUM=186597
FLUID_NUM=200000

class CubeObject:
    def __init__(self, minimum, size, id, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material[id]
