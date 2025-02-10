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

''' Configurações de Simulação '''

import math, torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg

@configclass
class ObservationsCfg:
    """ Especificações das Observações para o Ambiente """  

    @configclass
    class PolicyCfg(ObsGroup):
        """ Observações para Grupos de Politicas  """

        # Termos observados (ordem preservada)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) # ...
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel) # ...

        # ...
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # Grupos de Observação
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """ Especificações das Ações para o Ambiente """

    joint_efforts = mdp.JointEffortActionCfg(asset_name = "robot", 
                                             joint_names = ["slider_to_cart"],
                                             scale = 5.0)

@configclass
class EventCfg:
    """ Especificações dos Eventos para o Ambiente """

    # No inicio
    add_pole_mass = EventTerm(
        func = mdp.randomize_rigid_body_mass, 
        mode = "startup", 
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5), 
            "operation": "add",
        },
    )

    # No reset
    reset_cart_position = EventTerm(
        func = mdp.reset_joints_by_offset, 
        mode = "reset", 
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.1, 1.0), 
            "velocity_range": (-0.1, 0.1),
        },
    )

    # No reset
    reset_pole_position = EventTerm(
        func = mdp.reset_joints_by_offset, 
        mode = "reset", 
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.1 * math.pi, 1.0 * math.pi), 
            "velocity_range": (-0.1 * math.pi, 0.1 * math.pi),
        },
    )  


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """ configuração do Ambiente - Cartpole """

    # Confirações da Cena
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)

    # Configurações Básica -> Observações, Eventos e Ações
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        " Visualização: "
        # Configurações de Visualização
        self.viewer.eye = [4.5, 0.0, 6.0]     # Onde a camera está posicionada -> Olho do Observador
        self.viewer.lookat = [0.0, 0.0, 2.0]  # Onde a camera está olhando     -> Onde está o objeto desejado

        # Configuração do Passo: controla a frequência do ambiente (200 Hz / 4 = 50 Hz)
        self.decimation = 4  

        # Configuração da Simulação: tempo de passo para cada atualização (1/dt = 1/0.005 = 200 hz)
        self.sim.dt = 0.005  

def main():
    """ Função Main """
    
    # Instanciando os Argumentos e Configura o Ambiente do cartpole
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Configuração basica do Ambiente
    env = ManagerBasedEnv(cfg=env_cfg)

    # Simulanto a Física
    count = 0
    while simulation_app.is_running():
        # Reseta
        if count % 300 == 0:
            count = 0
            env.reset()
            print("-" * 80)
            print("[INFO] Resentando o Ambiente ...")

        # Ações simples randômicas
        joint_efforts = torch.randn_like(env.action_manager.action)
        # Step do Ambiente
        obs, _= env.step(joint_efforts)
        # Imprimi a oritação atual do cartpole
        print("[ENV 0]: Pole joint: ", obs["policy"][0][1].item())
        # Atualiza o contador
        count += 1
    
    # Fecha o Ambiente
    env.close()

if __name__ == "__main__":
    # Executando a função main
    main()
    # Fechando o App de Simulação
    simulation_app.close()
