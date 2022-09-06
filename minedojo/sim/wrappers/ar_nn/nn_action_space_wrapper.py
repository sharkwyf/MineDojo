import math
from typing import Union, Sequence

import gym
import numpy as np

from ...sim import MineDojoSim
from ....sim import spaces as spaces
from ....sim.mc_meta import mc as MC
from ....sim.inventory import InventoryItem


class NNActionSpaceWrapper(gym.Wrapper):
    """
    Action wrapper to transform native action space to a new space friendly to train NNs
    """

    def __init__(
        self,
        env: Union[MineDojoSim, gym.Wrapper],
        discretized_camera_interval: Union[int, float] = 15,
        strict_check: bool = True,
    ):
        assert (
            "inventory" in env.observation_space.keys()
        ), f"missing inventory from obs space"
        super().__init__(env=env)

        n_pitch_bins = math.ceil(360 / discretized_camera_interval) + 1
        n_yaw_bins = math.ceil(360 / discretized_camera_interval) + 1

        self.action_space = spaces.Dict(
            spaces={
                "ESC": spaces.Discrete(2),
                "attack": spaces.Discrete(2),
                "back": spaces.Discrete(2),
                "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
                "drop": spaces.Discrete(2),
                "forward": spaces.Discrete(2),
                "hotbar.1": spaces.Discrete(2),
                "hotbar.2": spaces.Discrete(2),
                "hotbar.3": spaces.Discrete(2),
                "hotbar.4": spaces.Discrete(2),
                "hotbar.5": spaces.Discrete(2),
                "hotbar.6": spaces.Discrete(2),
                "hotbar.7": spaces.Discrete(2),
                "hotbar.8": spaces.Discrete(2),
                "hotbar.9": spaces.Discrete(2),
                "inventory": spaces.Discrete(2),
                "jump": spaces.Discrete(2),
                "left": spaces.Discrete(2),
                "pickItem": spaces.Discrete(2),
                "right": spaces.Discrete(2),
                "sneak": spaces.Discrete(2),
                "sprint": spaces.Discrete(2),
                "swapHands": spaces.Discrete(2),
                "use": spaces.Discrete(2)
            },
            # noop_vec={
            #     "ESC":       0,
            #     "attack":    0,
            #     "back":      0,
            #     "camera.pitch": 0,
            #     "camera.yaw": 0,
            #     "drop":      0,
            #     "forward":   0,
            #     "hotbar.1":  0,
            #     "hotbar.2":  0,
            #     "hotbar.3":  0,
            #     "hotbar.4":  0,
            #     "hotbar.5":  0,
            #     "hotbar.6":  0,
            #     "hotbar.7":  0,
            #     "hotbar.8":  0,
            #     "hotbar.9":  0,
            #     "inventory": 0,
            #     "jump":      0,
            #     "left":      0,
            #     "pickItem":  0,
            #     "right":     0,
            #     "sneak":     0,
            #     "sprint":    0,
            #     "swapHands": 0,
            #     "use":       0,
            # },
        )
        self._cam_interval = discretized_camera_interval
        self._inventory_names = None
        self._strict_check = strict_check

    def action(self, action: Sequence[int]):
        """
        NN action to Malmo action
        """
        assert self.action_space.contains(action)
        destroy_item = (False, None)
        noop = self.env.action_space.no_op()

        # ------ parse main actions ------
        noop["attack"] = action["attack"]
        noop["back"] = action["back"]
        noop["camera"] = action["camera"]
        noop["drop"] = action["drop"]
        noop["forward"] = action["forward"]
        noop["inventory"] = action["inventory"]
        noop["jump"] = action["jump"]
        noop["left"] = action["left"]
        noop["right"] = action["right"]
        noop["sneak"] = action["sneak"]
        noop["sprint"] = action["sprint"]
        noop["use"] = action["use"]
        return noop, destroy_item

    def reverse_action(self, action):
        """
        Malmo action to NN action
        """
        # first convert camera actions to [-pi, +pi]
        action["camera"] = (
            np.arctan2(
                np.sin(action["camera"] * np.pi / 180),
                np.cos(action["camera"] * np.pi / 180),
            )
            * 180
            / np.pi
        )
        assert self.env.action_space.contains(action)

        noop = self.action_space.no_op()
        # ------ parse main actions ------
        # parse forward and back
        if action["forward"] == 1 and action["back"] == 1:
            # cancel each other, noop
            pass
        elif action["forward"] == 1:
            noop[0] = 1
        elif action["back"] == 1:
            noop[0] = 2
        # parse left and right
        if action["left"] == 1 and action["right"] == 1:
            # cancel each other, noop
            pass
        elif action["left"] == 1:
            noop[1] = 1
        elif action["right"] == 1:
            noop[1] = 2
        # parse jump, sneak, sprint
        # prioritize jump
        if action["jump"] == 1:
            noop[2] = 1
        else:
            if action["sneak"] == 1 and action["sprint"] == 1:
                # cancel each other, noop
                pass
            elif action["sneak"] == 1:
                noop[2] = 2
            elif action["sprint"] == 1:
                noop[2] = 3
        # parse camera pitch
        noop[3] = math.ceil((action["camera"][0] - (-180)) / self._cam_interval)
        # parse camera yaw
        noop[4] = math.ceil((action["camera"][1] - (-180)) / self._cam_interval)

        # ------ parse functional actions ------
        # order: attack > use > craft > equip > place > drop > destroy
        if action["attack"] == 1:
            noop[5] = 3
        elif action["use"] == 1:
            noop[5] = 1
        elif action["craft"] != "none" and action["craft"] != 0:
            craft = action["craft"]
            if isinstance(craft, int):
                craft = MC.ALL_PERSONAL_CRAFTING_ITEMS[craft - 1]
            noop[5] = 4
            noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(craft)
        elif action["craft_with_table"] != "none" and action["craft_with_table"] != 0:
            craft = action["craft_with_table"]
            if isinstance(craft, int):
                craft = MC.ALL_CRAFTING_TABLE_ITEMS[craft - 1]
            noop[5] = 4
            noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(craft)
        elif action["smelt"] != "none" and action["smelt"] != 0:
            smelt = action["smelt"]
            if isinstance(smelt, int):
                smelt = MC.ALL_SMELTING_ITEMS[smelt - 1]
            noop[5] = 4
            noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(smelt)
        elif action["equip"] != "none" and action["equip"] != 0:
            equip = action["equip"]
            if isinstance(equip, int):
                equip = MC.ALL_ITEMS[equip - 1]
            equip = equip.replace("_", " ")
            if equip not in self._inventory_names:
                if self._strict_check:
                    raise ValueError(
                        f"try to equip {equip}, but it is not in the inventory {self._inventory_names}"
                    )
            else:
                slot_idx = np.where(self._inventory_names == equip)[0][0]
                noop[5] = 5
                noop[7] = slot_idx
        elif action["place"] != "none" and action["place"] != 0:
            place = action["place"]
            if isinstance(place, int):
                place = MC.ALL_ITEMS[place - 1]
            place = place.replace("_", " ")
            if place not in self._inventory_names:
                if self._strict_check:
                    raise ValueError(
                        f"try to place {place}, but it is not in the inventory {self._inventory_names}"
                    )
            else:
                slot_idx = np.where(self._inventory_names == place)[0][0]
                noop[5] = 6
                noop[7] = slot_idx
        elif action["drop"] == 1:
            noop[5] = 2
        return noop

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._inventory_names = obs["inventory"]["name"].copy()
        return obs

    def step(self, action: Sequence[int]):
        malmo_action, destroy_item = self.action(action)
        destroy_item, destroy_slot = destroy_item
        if destroy_item:
            obs, reward, done, info = self.env.set_inventory(
                inventory_list=[
                    InventoryItem(name="air", slot=destroy_slot, quantity=1, variant=0)
                ],
                action=malmo_action,
            )
        else:
            obs, reward, done, info = self.env.step(malmo_action)

        # handle malmo's lags
        # if action[5] in {2, 4, 5, 6, 7}:
        for _ in range(1):
            obs, reward, done, info = self.env.step(self.env.action_space.no_op())
        self._inventory_names = obs["inventory"]["name"].copy()
        return obs, reward, done, info
