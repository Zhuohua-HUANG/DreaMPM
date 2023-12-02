import taichi as ti
import DreaMPM

ti.init(arch=ti.gpu)

config_path = 'config.yaml'

if __name__ == "__main__":
    dmpm = DreaMPM.DMPM(config_path)
    dmpm.run()
