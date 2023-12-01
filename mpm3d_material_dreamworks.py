import taichi as ti
import DreaMPM
ti.init(arch=ti.gpu)

# Hyperparameters
width, height = 1920, 1080
G_number, max_timestep, max_hard = 64, 2, 2.5  # more particles, num: 65536
# G_number, max_timestep, max_hard = 32, 4, 4  # less particles, num: 8192

config_path = './config.json'

if __name__ == "__main__":
    dmpm = DreaMPM.DMPM(width, height, G_number, max_timestep, max_hard, config_path)
    dmpm.run()
