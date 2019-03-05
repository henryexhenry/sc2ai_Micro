from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env
from pysc2 import maps
from absl import app
import random
import math
import numpy as np
map_01 = 'HK2V1_RESET_1DIE'
map_02 = 'FindAndDefeatZerglings'
_mapName = map_02

'''
Strategy:

射手vs近战
1. close-first
    - random search for discovering enemies
    - if found enemy, get enemies coordinates
    - compute distance
    - attack the closest enemy
    - for every ally:
        if distances(ally, enemy) < attack_range:
            ally moves back until reach attack_range
        else:
            attack
    code:
        if not find enemy:
            select all unit
            random move
        if found enemies:
            get enemies coor
            if distance(player unit, enemy) < attact range:
                select that unit
                move back
            else:
                find closest enemy
                select all unit
                attack enemy
2. low-hp-first
'''

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_ATTACK_RANGE = 12  # hy TODO

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_UNIT_ALLIANCE = 1
_UNIT_HEALTH = 2
_UNIT_X = 12
_UNIT_Y = 13
_UNIT_RADIUS = 15 # find range
_UNIT_HEALTH_RATIO = 7
_UNIT_IS_SELECTED = 17

_NOT_QUEUED = [0]
_QUEUED = [1]

DEFAULT_PLAYER_COUNT = 3
DEFAULT_ENEMY_COUNT = 1

class hyAgent(base_agent.BaseAgent):
    def __init__(self):
        super(hyAgent, self).__init__()
    
    def can_do(self, obs, action): # 判断动作是否可以被执行
        return action in obs.observation.available_actions # return True/False

    def extract_features(self, obs):
        var = obs.observation['feature_units']
        # get units' location and distance
        enemy, player = [], []

        # get health
        enemy_hp, player_hp = [], []

        # record the selected army
        is_selected = []

        # unit_count
        enemy_unit_count, player_unit_count = 0, 0

        for i in range(0, var.shape[0]):
            if var[i][_UNIT_ALLIANCE] == _PLAYER_HOSTILE:
                # find enemy units
                enemy.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                enemy_hp.append(var[i][_UNIT_HEALTH])
                enemy_unit_count += 1
            else:
                # find player units
                player.append((var[i][_UNIT_X], var[i][_UNIT_Y]))
                player_hp.append(var[i][_UNIT_HEALTH])
                is_selected.append(var[i][_UNIT_IS_SELECTED])
                player_unit_count += 1
        '''
        # append if necessary
        for i in range(player_unit_count, DEFAULT_PLAYER_COUNT):
            player.append((-1, -1))
            player_hp.append(0)
            is_selected.append(-1)

        for i in range(enemy_unit_count, DEFAULT_ENEMY_COUNT):
            enemy.append((-1, -1))
            enemy_hp.append(0)
        '''    
        # get distance
        min_distance = [100000 for x in range(DEFAULT_PLAYER_COUNT)]

        for i in range(0, player_unit_count):
            for j in range(0, enemy_unit_count):
                distance = int(math.sqrt((player[i][0] - enemy[j][0]) ** 2 + (
                        player[i][1] - enemy[j][1]) ** 2))

                if distance < min_distance[i]:
                    min_distance[i] = distance

        # flatten the array so that all features are a 1D array
        enemy_hp_fltn = np.array(enemy_hp).flatten() # enemy's hp
        player_hp_fltn = np.array(player_hp).flatten() # player's hp
        enemy_fltn = np.array(enemy).flatten() # enemy's coordinates
        player_fltn = np.array(player).flatten() # player's coordinates
        min_distance_fltn = np.array(min_distance).flatten() # distance

        if player and enemy:
            closest_coor, enemy_coor, unit_index, enemy_index, dist = self.closest_enemy(player, enemy)

        '''
        selecting_closest = []
        index = -1
        
        for i in range(0, player_unit_count):
            if is_selected[i] == 1 and is_selected[1-i] != 1:
                index = i
        if index == unit_index :
            selecting_closest.append(1)
        else:
            selecting_closest.append(0)
        '''
        # combine all features horizontally
        current_state = np.hstack((enemy_fltn, player_fltn, is_selected))
        print("is_selected = ", is_selected)

        return current_state, enemy_hp_fltn, player_hp_fltn, enemy, player, min_distance, is_selected, enemy_unit_count, player_unit_count

    def get_units_by_type(self, obs, unit_type):  # 通过判断类型 从而选取单位
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def unit_type_is_selected(self, obs, unit_type):

        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def calculate_distance(self, single_unit_coor, single_enemy_coor):
        dist = int(math.sqrt((single_unit_coor[0] - single_enemy_coor[0]) ** 2 + (single_unit_coor[1] - single_enemy_coor[1]) ** 2))
        return dist

    def closest_enemy(self, unit_locs, enemy_locs):
        dist = 10000
        index = -1
        unit_index = 0
        enemy_index = 0
        #for i in range(0, 2):
        for i in range(len(unit_locs)):
            for j in range(len(enemy_locs)):
                if self.calculate_distance(unit_locs[i], enemy_locs[j]) < dist:
                    dist = self.calculate_distance(unit_locs[i], enemy_locs[j])
                    unit_index = i
                    enemy_index = j
        return unit_locs[unit_index], enemy_locs[enemy_index], unit_index, enemy_index, dist

    def move_backward(self, unit_coor, enemy_coor):
        if enemy_coor[0] - unit_coor[0] >= 0 :
            if enemy_coor[1] - unit_coor[1] >= 0:
                # ++
                x = unit_coor[0] - _ATTACK_RANGE//2
                y = unit_coor[1] - _ATTACK_RANGE//2
                
            else:
                # +-
                x = unit_coor[0] - _ATTACK_RANGE//2
                y = unit_coor[1] + _ATTACK_RANGE//2
        else:
            if enemy_coor[1] - unit_coor[1] >= 0:
                # -+
                x = unit_coor[0] + _ATTACK_RANGE//2
                y = unit_coor[1] - _ATTACK_RANGE//2
            else:
                # --
                x = unit_coor[0] + _ATTACK_RANGE//2
                y = unit_coor[1] + _ATTACK_RANGE//2
        # avoid out of range error
        if x > 80:
            x = 80
        elif x < 8:
            x = 8
        if y > 80:
            y = 80
        elif y < 8:
            y = 8
        return x, y



    def step(self, obs):
        super(hyAgent, self).step(obs)

        # extract features
        current_state, enemy_hp, player_hp, enemy_loc, player_loc, distance, selected, enemy_count, player_count = self.extract_features(obs)

        # if no enemy in the screen
        # attack minimap at coor (49,49)
        if enemy_count == 0:
            self.attack_coordinates = (random.randint(0,49), random.randint(0,49))

            if self.unit_type_is_selected(obs, units.Terran.Marine):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap([1], self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
        else: # enemy count > 0
            if player_loc:
                unit_coor, enemy_coor, unit_index, enemy_index, dist = self.closest_enemy(player_loc, enemy_loc)

                # if enemy is too close to unit
                if dist < _ATTACK_RANGE // 2:
                    # move backward
                    '''
                    x, y = self.move_backward(unit_coor, enemy_coor)
                    if self.unit_type_is_selected(obs, units.Terran.Marine):
                        if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                            return actions.FUNCTIONS.Move_screen([0], [x, y])
                    '''
                    # instead of moving the whole team backward, i want to only move the closest unit back.
                    # if the closest unit has been selected, 
                    x, y = self.move_backward(unit_coor, enemy_coor)
                    if len(obs.observation.single_select) == 1: 
                    # if self.unit_type_is_selected(obs, units.Terran.Marine):
                        if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                            return actions.FUNCTIONS.Move_screen([0], [x, y])

                    if self.can_do(obs, actions.FUNCTIONS.select_unit.id):
                        return actions.FUNCTIONS.select_unit("select", [unit_index])

                if self.unit_type_is_selected(obs, units.Terran.Marine):
                    if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                        _x, _y = enemy_coor
                        if enemy_coor[0] > 80:
                            _x = 80
                        if enemy_coor[0] < 8:
                            _x = 8
                        if enemy_coor[1] > 80:
                            _y = 80
                        if enemy_coor[1] < 8:
                            _y = 8
                        return actions.FUNCTIONS.Attack_screen([1], [_x, _y])


            '''
            if obs.first():
                # nonzero 返回2d list中非零元素的坐标
                player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
                xmean = player_x.mean()
                ymean = player_y.mean()
            
                if xmean <= 31 and ymean <= 31:
                    self.attack_coordinates = (49, 49)
                else:
                    self.attack_coordinates = (12, 16)
            '''

            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

def main(unused_argv):
    agent = hyAgent()
    '''
    globals()[_mapName] = type(
        _mapName, (maps.mini_games.MiniGame,), dict(filename=_mapName))
    '''

    try:
        while True:
            with sc2_env.SC2Env(
                    map_name=_mapName,
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=84, minimap=64),  # 设置画面解像度，决定机器人的视野
                        use_feature_units=True
                    ),
                    step_mul=2,  # 机器人采取行动（action）前需要进行的‘game step’ 数量 （数值越小，APM越高）
                    game_steps_per_episode=0,  # 游戏时长，设为0即不限时
                    visualize=True) as env:  # optional 显示出observation layer 的detail， 帮助我们理解机器人的进程

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset() # start a new episode, update _obs
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)