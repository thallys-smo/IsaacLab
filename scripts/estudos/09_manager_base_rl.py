''' Inicializando o Simulador '''

import argparse
from isaaclab.app import AppLauncher

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
            "asset_cfg": SceneEntityCfg("robot", joint_names= ["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5), 
        }
    )

    reset_pole_position = EventTerm(
        func = mdp.reset_joints_by_offset, 
        mode = "reset", 
        params= {
            "asset_cfg": SceneEntityCfg("robot", joint_names = ["cart_to_pole"]), 
            "position_range": (-0.25 * math.pi, 0.25 * math.pi), 
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)
        },
    )
    
@configclass
class RewardsCfg:
    ''' Termos de Recompensas para o MDP '''

    # (1): Recompensa de Vida
    alive = RewTerm(func = mdp.is_alive, weight = 1.0)

    # (2): Penalidade de falha
    terminating = RewTerm(func = mdp.is_terminated, weight= - 2.0)

    # (3): Tarefa Principal: Manter o Pole para cima
    pole_pos = RewTerm(
        func = mdp.joint_pos_target_l2, 
        weight = - 1.0,
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names = ["cart_to_pole"]), "target": 0.0}
    )

    # (4): Shapping Tasks: Baixa velocidade do carrinho
    cart_vel = RewTerm(
        func = mdp.joint_vel_l1, 
        weight= - 0.01, 
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names = ["slider_to_cart"])
        },
    )

    # (5): Shapping Tasks: Baixa velocidade angulação do pole
    pole_vel = RewTerm(
        func = mdp.joint_vel_l1, 
        weight= - 0.005, 
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names = ["slider_to_cart"])
        },
    )

@configclass
class TerminationsCfg:
    ''' Termos de Rescisão do MDP  '''

    # (1): Tempo Esgotado
    time_out = DoneTerm(func=mdp.time_out, time_out= True)

    # (2): Carro fora dos limites
    cart_out_of_bounds = DoneTerm(
        func = mdp.joint_pos_out_of_manual_limit, 
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names = ["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    ''' Configuração do Ambiente - Cartpole '''

    # Configurações de Cena
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs = 4096, env_spacing = 4.0)

    # Configurações Basicas
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # Configurações do MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Inicialização de Posição
    def __post_init__(self) -> None:
        ''' Inicialização de Posição '''
        
        # Configurações Gerais
        self.decimation = 2
        self.episode_length_s = 5

        # Configurações de Visualização
        self.viewer.eye = (8.0, 0.0, 5.0)

        # Configurações de Simulação
        self.sim.dt = 1/120
        self.sim.render_interval = self.decimation

def main():
    ''' Função Main '''
    
    # Instaciando os Argumentos e Configuração do Ambiente do Cartpole
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

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
