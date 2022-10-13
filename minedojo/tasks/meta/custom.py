from typing import Union, Optional, List, Dict, Tuple, Any

from minedojo.tasks.meta.utils import reward_fns, success_criteria

from .base import MetaTaskBase
from ...sim.inventory import InventoryItem
from .utils import survive_per_day_reward, survive_n_days_reward, time_since_death_check,simple_inventory_based_check
from .extra_spawn import SpawnItem2Condition

from .utils import (
    always_satisfy_condition,
    simple_stat_kill_entity_based_check,
    simple_stat_kill_entity_based_reward,
)
import numpy as np
import gym

by_setblock = ["diamond_ore", "gold_ore", "iron_ore", "coal_ore","crafting_table"]
by_summon = ["pig", "cow", "bat", "cat", "chicken", "horse", "sheep", "zombie","spider","witch","rabbit"]

class CustomMeta(MetaTaskBase):
    """
    Class for survival tasks.
    Args:
        break_speed_multiplier: Controls the speed of breaking blocks. A value larger than 1.0 accelerates the breaking.
                Default: ``1.0``.

        event_level_control: If ``True``, the agent is able to perform high-level controls including place and equip.
                If ``False``, then is keyboard-mouse level control.
                Default: ``True``.

        fast_reset: If ``True``, reset using MC native command `/kill`, instead of waiting for re-generating new worlds.
            Default: ``True``.

            .. warning::
                Side effects:

                1. Changes to the world will not be reset. E.g., if the agent chops lots of trees then calling
                fast reset will not restore those trees.

                2. If you specify agent starting health and food, these specs will only be respected at the first reset
                (i.e., generating a new world) but will not be respected in the following resets (i.e., reset using MC cmds).
                So be careful to use this wrapper if your usages require specific initial health/food.

                3. Statistics/achievements will not be reset. This wrapper will maintain a property ``info_prev_reset``.
                If your tasks use stat/achievements to evaluation, please retrieve this property and compute differences.

        image_size: The size of image observations.

        initial_weather: If not ``None``, specifies the initial weather.
                Can be one of ``"clear"``, ``"normal"``, ``"rain"``, ``"thunder"``.
                Default: ``None``.

        lidar_rays: Defines the directions and maximum distances of the lidar rays if ``use_lidar`` is ``True``.
                If supplied, should be a list of tuple(pitch, yaw, distance).
                Pitch and yaw are in radians and relative to agent looking vector.
                Default: ``None``.

        per_day_reward: The reward value for each day of survival
                Default: ``1``.

        seed: The seed for an instance's internal generator.
                Default: ``None``.

        sim_name: Name of a simulation instance.
                Default: ``"SurvivalMeta"``.

        start_food: If not ``None``, specifies initial food condition of the agent.
                Default: ``None``.

        start_health: If not ``None``, specifies initial health condition of the agent.
                Default: ``None``.

        start_position: If not ``None``, specifies the agent's initial location and orientation in the minecraft world.
                Default: ``None``.

        success_reward: The final reward value if survived target days.
                Default: ``None``.

        target_days: Target days to survive.

        use_lidar: If ``True``, includes lidar in observations.
                Default: ``False``.

        use_voxel: If ``True``, includes voxel in observations.
                Default: ``False``.

        voxel_size: Defines the voxel's range in each axis if ``use_voxel`` is ``True``.
                If supplied, should be a dict with keys ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
                Each value specifies the voxel size relative to the agent.
                Default: ``None``.

        world_seed: Seed for deterministic world generation
                if ``generate_world_type`` is ``"default"`` or ``"specified_biome"``.
                See `here <https://minecraft.fandom.com/wiki/Seed_(level_generation)>`_ for more details.
                Default: ``None``.
    """


    # https://minecraft.fandom.com/el/wiki/Day-night_cycle

    def __init__(
        self,
        *,
        # ------ target days and reward per day
        # target_days: int,
        # per_day_reward: Union[int, float] = 1,
        target_names: Union[str, List[str]] = None,
        target_quantities: Union[int, List[int], Dict[str, int]] = None,
        reward_weights: Union[int, float, Dict[str, Union[int, float]]] = None,
        # ------ initial & extra spawn control ------
        initial_mobs: Optional[Union[str, List[str]]] = None,
        initial_mob_spawn_range_low: Optional[Tuple[int, int, int]] = None,
        initial_mob_spawn_range_high: Optional[Tuple[int, int, int]] = None,
        min_spawn_range : Optional[int] = None,
        initial_blocks: Optional[Union[str, List[str]]] = None,
        initial_block_set_range_low: Optional[Tuple[int, int, int]] = None,
        initial_block_set_range_high: Optional[Tuple[int, int, int]] = None,
        min_block_range : Optional[int] = None,
        spawn_rate: Optional[
            Union[float, int, List[Union[float, int]], Dict[str, Union[float, int]]]
        ] = None,
        spawn_range_low: Optional[Tuple[int, int, int]] = None,
        spawn_range_high: Optional[Tuple[int, int, int]] = None,
        # ------ initial conditions ------
        start_position: Optional[Dict[str, Union[float, int]]] = None,
        start_health: Optional[float] = None,
        start_food: Optional[int] = None,
        start_at_night: bool = None,
        initial_weather: Optional[str] = None,
        initial_inventory: Optional[List[InventoryItem]] = None,
        drawing_str: Optional[str] = None,
        # ------ global conditions ------
        specified_biome: Optional[Union[int, str]] = None,
        always_night: bool = None,
        allow_mob_spawn: bool = None,
        break_speed_multiplier: float = 1.0,
        # ------ sim seed & world seed ------
        seed: Optional[int] = None,
        world_seed: Optional[str] = None,
        # ------ reset mode ------
        fast_reset: bool = True,
        fast_reset_random_teleport_range: Optional[int] = None,
        # ------ obs ------
        image_size: Union[int, Tuple[int, int]],
        use_voxel: bool = False,
        voxel_size: Optional[Dict[str, int]] = None,
        use_lidar: bool = False,
        lidar_rays: Optional[List[Tuple[float, float, float]]] = None,
        # ------ event-level action or keyboard-mouse level action ------
        event_level_control: bool = True,
        success_reward: Optional[Union[int, float]] = None,
        guidance: str = None,
        task: str = None,
    ):
        target_days = 1
        per_day_reward = 1
        self._target_days = target_days
        success_reward = success_reward or target_days * per_day_reward
        mc_ticks_per_day = 24000

        reward_fns = [
            survive_per_day_reward(mc_ticks_per_day, per_day_reward),
            survive_n_days_reward(mc_ticks_per_day, target_days, success_reward),
        ]
        success_criteria = [time_since_death_check(mc_ticks_per_day * target_days)]

        self._extra_spawn_rate = None
        self.min_spawn_range = min_spawn_range
        self.min_block_range = min_block_range
        if spawn_rate is not None:
            assert all(
                [1.0 >= rate >= 0.0 for rate in spawn_rate.values()]
            ), "extra spawn rate must <= 1.0 and >= 0.0"
            assert all(
                [
                    name in self.by_summon or name in self.by_setblock
                    for name in spawn_rate.keys()
                ]
            ), f"{spawn_rate.keys()} should belong to either {self.by_summon} or {self.by_setblock}"
            if extra_spawn_condition is None:
                extra_spawn_condition = {
                    k: always_satisfy_condition for k in spawn_rate.keys()
                }
            else:
                if isinstance(extra_spawn_condition, dict):
                    assert set(spawn_rate.keys()) == set(
                        extra_spawn_condition.keys()
                    ), (
                        f"extra_spawn_rate must match extra_spawn_condition, "
                        f"but got {spawn_rate.keys()} and {extra_spawn_condition.keys()}"
                    )
                else:
                    extra_spawn_condition = {
                        k: extra_spawn_condition for k in spawn_rate.keys()
                    }
            self._extra_spawn_rate = spawn_rate
            self._extra_spawn_rates_and_conditions = {
                k: (spawn_rate[k], extra_spawn_condition[k])
                for k in spawn_rate.keys()
            }
            assert (
                spawn_range_high is not None and spawn_range_low is not None
            )
            assert len(spawn_range_high) == 3 and len(spawn_range_low) == 3
            low = np.repeat(
                np.array(spawn_range_low)[np.newaxis, ...],
                len(spawn_rate),
                axis=0,
            )
            high = np.repeat(
                np.array(spawn_range_high)[np.newaxis, ...],
                len(spawn_rate),
                axis=0,
            )
            self._extra_spawn_range_space = gym.spaces.Box(
                low=low, high=high, seed=seed
            )
            self._rng = np.random.default_rng(seed=seed)


        if isinstance(target_names, str):
            target_names = [target_names]
        if isinstance(target_quantities, int):
            target_quantities = {k: target_quantities for k in target_names}
        elif isinstance(target_quantities, list):
            assert len(target_names) == len(target_quantities)
            target_quantities = {
                k: target_quantities[i] for i, k in enumerate(target_names)
            }
        elif isinstance(target_quantities, dict):
            assert set(target_names) == set(target_quantities.keys())
        if isinstance(reward_weights, int) or isinstance(reward_weights, float):
            reward_weights = {k: reward_weights for k in target_names}
        elif isinstance(reward_weights, dict):
            assert set(target_names) == set(reward_weights.keys())

        if target_quantities is not None:
            self._target_quantities = target_quantities

        spawn_condition = None
        if spawn_rate is not None:
            if isinstance(spawn_rate, float) or isinstance(spawn_rate, int):
                spawn_rate = {k: spawn_rate for k in target_names}
            elif isinstance(spawn_rate, list):
                assert len(spawn_rate) == len(target_names)
                spawn_rate = {k: spawn_rate[i] for i, k in enumerate(target_names)}
            elif isinstance(spawn_rate, dict):
                # don't do any checks here, so users can specify arbitrary spawn rates
                pass
            spawn_condition = {
                k: SpawnItem2Condition.get(k, always_satisfy_condition)
                for k in spawn_rate.keys()
            }

        initial_blocks = initial_blocks or []
        if isinstance(initial_blocks, str):
            initial_blocks = [initial_blocks]
        self._initial_blocks = initial_blocks

        if len(initial_blocks) > 0:
            assert len(initial_block_set_range_low) == 3
            assert len(initial_block_set_range_high) == 3
            low = np.repeat(
                np.array(initial_block_set_range_low)[np.newaxis, ...],
                len(initial_blocks),
                axis=0,
            )
            high = np.repeat(
                np.array(initial_mob_spawn_range_high)[np.newaxis, ...],
                len(initial_blocks),
                axis=0,
            )
            low = np.array(initial_block_set_range_low)
            high = np.array(initial_block_set_range_high)
            self._block_set_range_space = gym.spaces.Box(
                low=low, high=high, seed=seed
            )

        initial_mobs = initial_mobs or []

        if isinstance(initial_mobs, str):
            initial_mobs = [initial_mobs]
        self._initial_mobs = initial_mobs
        if len(initial_mobs) > 0:
            assert len(initial_mob_spawn_range_low) == 3
            assert len(initial_mob_spawn_range_high) == 3
            low = np.repeat(
                np.array(initial_mob_spawn_range_low)[np.newaxis, ...],
                len(initial_mobs),
                axis=0,
            )
            high = np.repeat(
                np.array(initial_mob_spawn_range_high)[np.newaxis, ...],
                len(initial_mobs),
                axis=0,
            )
            low = np.array(initial_mob_spawn_range_low)
            high = np.array(initial_mob_spawn_range_high)
            self._mob_spawn_range_space = gym.spaces.Box(
                low=low, high=high, seed=seed
            )

        generate_world_type = (
            "default" if specified_biome is None else "specified_biome"
        )

        start_time = 18000 if start_at_night else None

        super().__init__(
        #     initial_mobs=initial_mobs,
        #     initial_mob_spawn_range_low=initial_mob_spawn_range_low,
        #     initial_mob_spawn_range_high=initial_mob_spawn_range_high,
        #     min_spawn_range=min_spawn_range,
        #     extra_spawn_rate=spawn_rate,
        #     extra_spawn_condition=spawn_condition,
        #     extra_spawn_range_low=spawn_range_low,
        #     extra_spawn_range_high=spawn_range_high,
            fast_reset=fast_reset,
            success_criteria=success_criteria,
            reward_fns=reward_fns,
            seed=seed,
            image_size=image_size,
            use_voxel=use_voxel,
            voxel_size=voxel_size,
            use_lidar=use_lidar,
            lidar_rays=lidar_rays,
            event_level_control=event_level_control,
            initial_inventory=initial_inventory,
            break_speed_multiplier=break_speed_multiplier,
            world_seed=world_seed,
            start_position=start_position,
            initial_weather=initial_weather,
            start_health=start_health,
            start_food=start_food,
            generate_world_type=generate_world_type,
            specified_biome=specified_biome,
        )

    def step(self, action):
        if self._extra_spawn_rate is None:
            return super().step(action=action)
        else:
            rel_positions = self._extra_spawn_range_space.sample()
            for (name, (rate, condition)), pos in zip(
                self._extra_spawn_rates_and_conditions.items(), rel_positions
            ):
                if self._rng.random() <= rate and condition(
                    ini_info_dict=self._ini_info_dict,
                    pre_info_dict=self._pre_info_dict,
                    elapsed_timesteps=self._elapsed_timesteps,
                ):
                    if name in self.by_summon:
                        obs, _, _, info = self.env.spawn_mobs(name, pos)
                    elif name in self.by_setblock:
                        obs, _, _, info = self.env.set_block(name, pos)
            return super().step(action=action)


    def _after_sim_reset_hook(
        self, reset_obs: Dict[str, Any], reset_info: Dict[str, Any]
    ) -> (Dict[str, Any], Dict[str, Any]):
        obs, info = reset_obs, reset_info
        mobs_rel_positions = []
        if len(self._initial_mobs) > 0:
            for i in range(len(self._initial_mobs)):
                mobs_rel_position = self._mob_spawn_range_space.sample()
                if self.min_spawn_range != None:
                    dis = mobs_rel_position[0]**2 + mobs_rel_position[2]**2
                    print(dis)
                    while dis < self.min_spawn_range**2:
                        print('retry')
                        mobs_rel_position = self._mob_spawn_range_space.sample()
                        dis = mobs_rel_position[0]**2 + mobs_rel_position[1]**2
                mobs_rel_positions.append(mobs_rel_position)
                print('mobs_rel_position: ',mobs_rel_position)

            print('mobs_rel_positions: ',mobs_rel_positions)
            obs, _, _, info = self.env.spawn_mobs(
                self._initial_mobs, mobs_rel_positions
            )
        blocks_rel_positions = []
        if len(self._initial_blocks) > 0:
            for i in range(len(self._initial_blocks)):
                blocks_rel_position = self._block_set_range_space.sample()
                if self.min_block_range != None:
                    dis = blocks_rel_position[0]**2 + blocks_rel_position[2]**2
                    print(dis)
                    while dis < self.min_block_range**2:
                        print('retry')
                        blocks_rel_position = self._block_set_range_space.sample()
                        dis = blocks_rel_position[0]**2 + blocks_rel_position[1]**2
                blocks_rel_positions.append(blocks_rel_position)
                print('blocks_rel_position: ',blocks_rel_position)

            print('blocks_rel_positions: ',blocks_rel_positions)
            # obs, _, _, info = self.env.set_block(
            #     self._initial_blocks, blocks_rel_positions
            # )
            
            obs, _, _, info = self.env.set_block(
                self._initial_blocks, [[0,0,1]]
            )
            print('finish setting blocks')

        return obs, info

    @property
    def task_prompt(self) -> str:
        if self._target_quantities is not None:
            filling = ", ".join(
                [
                    f"{v} {str(k).replace('_', ' ')}"
                    for k, v in self._target_quantities.items()
                ]
            )
            return super().get_prompt(targets=filling)
        return super().get_prompt(target=self._target_days)
