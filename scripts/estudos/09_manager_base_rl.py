''' Inicializando o Simulador '''

import argparse
import isaaclab.app import AppLauncher

# Adição de Argumentos de Argpaser

parser = argparse.ArgumentParser(description = "Tutorial sobre como usar um Ambiente Base")
parser.add_argument("--num_envs", type=int, default=16, help= "Números de Ambientes Gerados")

# Acrescentando os Argumentos ao CLI do AppLauncher
AppLauncher.add_app_launcher_args(parser)
# Instanciando os Argumentos
args_cli = parser.parse_args()

# Inicializando o Aplicativo do Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

''' Configuração de Simulação '''

import math, torch


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

# Definicições Pré configuradas
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# Configuração de Cena

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ''' Configuração para Cena do Cartpole '''

    # Chão
    ground = AssetBaseCfg(
        prim_path = "/World/ground",
        spawn = sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Cartpole
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path = "{ENV_REGEX_NS}/Robot")

    # Luzes
    dome_light = AssetBaseCfg(
        prim_path = "/World/DomeLight",
        spawn = sim_utils.DomeLightCfg(color = (0.9, 0.9, 0.9), intensity = 500.0),
    )

# MDP Configurações

@configclass
class ActionsCfg:
    ''' Especificação da Ação para o MDP '''

    joint_effort = mdp.JointEffortActionCfg(
        asset_name = "robot",
        joint_names = ["slider_to_cart"], 
        scale = 100.0    
    )

@configclass
class ObservationsCfg:
    ''' Especificação da Observação para o MDP '''

    @configclass
    class PolicyCfg(ObsGroup):
        ''' Observações para o Grupo da Politica '''

        # Termos observados (ordem preservada)
        joint_pol_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Grupos de Observação
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    ''' Configuração de Eventos '''

    # No resete
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode= "reset",
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names= [ "slidet_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5                                
                               ), 
        }
    )


@configclass
class RewardsCfg:
    ''''''

@configclass
class TerminationsCfg:
    ''''''

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    ''''''

def main():
    ''' Função Main '''
    
    # Instaciando os Argumentos e Configuração do Ambiente do Cartpole
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.nun_envs

    # Configurando o Ambiente RL
    env = ManagerBasedRLEnv(env_cfg)

    # Simulando a Fisica
    count = 0 
    while(simulation_app.is_running()):
        # Reseta
        if count % 500 == 0:
            count = 0
            env.reset()   
            print("-" * 80)
            print("[INFO] Resentando o Ambiente ...")
        
        # Ação simples randomica
        joint_effords = torch.randn_like(env.action_manager.action)

        # Step o ambiente
        obs, rew, terminated, truncated, info = env.step(joint_effords)

        # Imprimi a Orientação da Politica Atual
        print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())

        # Atualiza o contador
        count += 1

    # Fecha o ambiente
    env.close()

if __name__ == "__main__": 
    # Executando a Função Main 
    main()
    # Fechando a simulação
    simulation_app.close()                             
