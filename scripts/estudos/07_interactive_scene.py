''' Inicializando o Simulador'''

import argparse

from isaaclab.app import AppLauncher

# Adição dos Argumentos de Argparse
parser = argparse.ArgumentParser(description=  "Tutorial sobre como usar a Interface de Cena Interativa")
parser.add_argument("--num_envs", type = int, default = 2, help = "Number de ambientes a gerar")

# Acrescentar Argumentos CLI do AppLauncher
AppLauncher.add_app_launcher_args(parser)
# Analisar os Argumentos - Não acho que esteja certo, mas usaremos isso por enquanto
args_cli = parser.parse_args()

# Iniciar o Aplicatio do Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

''' Configuracao da Simulacao '''

# Bibliotecas Associadas

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


# Importando Pre-Configuracoes -> CARTPOLE
from isaaclab_assets import CARTPOLE_CFG 

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
  """ Configuracao de uma Cena utilizando o CARTPOLE """

  # Chao
  ground = AssetBaseCfg(prim_path = "/World/defaultGroundPlane", 
                        spawn = sim_utils.GroundPlaneCfg())
  
  # Luzes
  dome_light = AssetBaseCfg(prim_path = "/World/Light", 
                            spawn = sim_utils.DomeLightCfg(intensity = 3000.0, color = (0.75, 0.75, 0.75)))
  
  # Articulacao
  cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path = "{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

  # Executa o Simulador em Loop

  robot = scene["cartpole"]

  # Step da Simulacao
  sim_dt = sim.get_physics_dt()
  count = 0

  # Loop de Simulacao
  while simulation_app.is_running():
      # Reset
      if count % 500 == 0: 
          # Reseta o contador
          count = 0
          """ A ideia do código abaixo é configurar o zero dos ambientes: 
          Imagine 1000 robos, cada um esta 1m do outro, se não fizer o que está abaixo
          eles todos setrão inicializados a 1m do zero do ambiente, não um do outro."""
          root_state = robot.data.default_root_state.clone()
          root_state[:, :3] += scene.env_origins
          robot.write_root_pose_to_sim(root_state[:, :7])
          robot.write_root_velocity_to_sim(root_state[:, 7:])

          # Defini as posições da juntas com algum ruido
          joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
          joint_pos += torch.rand_like(joint_pos) * 0.1
          robot.write_joint_state_to_sim(joint_pos, joint_vel)

          # Limpando o Buffers internos
          scene.reset()
      
      # Aplicando uma ação randomica

      # Gerando uma força randomica na junta
      efforts = torch.randn_like(robot.data.joint_pos) * 5.0

      # Aplicando a ação no robo
      robot.set_joint_effort_target(efforts)

      # Escrevendo os dados na Simulação
      scene.write_data_to_sim()

      # Executando um Step
      sim.step()

      # Incrementando o contador
      count += 1

      # Atualizando Buffers
      scene.update(sim_dt)

def main():
  """ Funcao Main """
  sim_cfg = sim_utils.SimulationCfg(device = args_cli.device)
  sim = SimulationContext(sim_cfg)

  # Configuracao da Camera
  sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

  # Cena -> Utilizou uma cena pronta na biblioteca, CARTPOLE
  scene_cfg = CartpoleSceneCfg(num_envs = args_cli.num_envs, env_spacing = 2.0)
  scene = InteractiveScene(scene_cfg)

  # Inicia o Simulador
  sim.reset()

  # Imprimindo que está configurado e pronto para rodar
  print("[INFO]: Configuração completa.")

  # Executa o Simulador
  run_simulator(sim, scene)

# Main

if __name__ == "__main__":
  # Executa a funcao Main
  main()
  # Fecha o Aplciativo SIM
  simulation_app.close()




