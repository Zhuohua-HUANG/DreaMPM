class CubeObject:
    def __init__(self, minimum, size, id, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material[id]
