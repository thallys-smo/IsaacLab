''' Inicializando o Simulado '''

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    ''' Configurações Especificas do Ambiente Cartpole '''

    # Ambiente
    decimation = 2
    episode_length_s = 5.0   # Duração de cada episodia em [s]
    action_scale = 100.0     # Escala aplicação às ações, representando a força em [N]
    action_space = 1         # Dimensão do Espaço de Estado (neste caso, 1)
    observation_space = 4    # Dimensão do Espaço de Observação (neste caso, 4)
    state_space = 0          # Dimensão do Espaço de Estado (não utilizado, pois é 0)

    # Simulação
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    
    # Robo
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path = "/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart" # Grau de liberdade do carrinho
    pole_dof_name = "cart_to_pole"   # Grau de liberdade do pêndulo

    # Cena
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Reset
    max_cart_pos = 3.0  # Posição máxima permitida para o carrinho antes de uma reinicialização
    initial_pole_angle_range = [-0.25, 0.25] #  Intervalo de ângulos iniciais para o pêndulo durante a reinicialização.

    # Escala das Recompensas
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = - 1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = - 0.005

class CartpoleEnv(DirectRLEnv):
    ''' Lógica do Ambiente de Treinamento Cartpole '''
    cfg: CartpoleEnvCfg

    def __init__(self, cfg:CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        ''' '''

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        # Armazena as posições e velocidades das juntas
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        ''' '''

        self.cartpole = Articulation(self.cfg.robot_cfg)

        # Adicionando o Chão
        spawn_ground_plane(prim_path= "/World/ground", cfg = GroundPlaneCfg())

        # Clona e Replica o que? 
        self.scene.clone_environments(copy_from_source = False)

        # Adiciona uma Articulação na Cena
        self.scene.articulations["cartpole"] = self.cartpole

        # Adiciona Luzes
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color = (0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            ''' '''

            self.actions = self.action_scale * actions.clone()

        def _apply_action(self) -> None: 
            ''' '''
            self.cartpole.set_joint_effort_target(self.actions, joints_ids = self._cart_dof_idx)

        def _get_observations(self) -> dict: 
            ''' '''
            obs = torch.cat(
                (
                    self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            )
            observations = {"policy", obs}
            return observations

        def _get_rewards(self) -> Torch.Tensor: 
            ''' '''
            total_rewards = compute_rewards(
                self.cfg.rew_scale_alive,
                self.cfg.rew_scale_terminated,
                self.cfg.rew_scale_pole_pos,
                self.cfg.rew_scale_cart_vel,
                self.cfg.rew_scale_pole_vel,
                self.joint_pos[:, self._pole_dof_idx[0]],
                self.joint_vel[:, self._pole_dof_idx[0]],
                self.joint_pos[:, self._cart_dof_idx[0]],
                self.joint_vel[:, self._cart_dof_idx[0]],
                self.reset_terminated,
            )  
            return total_rewards
        
        
         





