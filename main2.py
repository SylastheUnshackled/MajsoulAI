# -*- coding: utf-8 -*-
import json
import time
import select
import socket
import pickle
import os
import argparse
import importlib
from typing import Dict, List, Tuple
from urllib.parse import quote, unquote
from enum import Enum
from xmlrpc.client import ServerProxy
from subprocess import Popen, CREATE_NEW_CONSOLE
import numpy as np

import majsoul_wrapper as sdk
from majsoul_wrapper import all_tiles, Operation


class State(Enum):  # 控制AI进程与Majsoul进程同步
    WaitingForStart = 0
    Playing = 1

def majsoul2mjai(tile: str) -> str:
    """
    将雀魂的牌表示转换为 mjai 格式
    支持普通牌、字牌和红宝牌
    """
    # 字牌转换字典
    majsoul_to_mjai_dict = {
        '1z': 'E',  # 东风
        '2z': 'S',  # 南风
        '3z': 'W',  # 西风
        '4z': 'N',  # 北风
        '5z': 'P',  # 白板
        '6z': 'F',  # 发财
        '7z': 'C'   # 中
    }
    
    # 红宝牌转换
    if tile[0] == '0':
        return '5' + tile[-1] + 'r'
    
    # 字牌转换
    if tile[-1] == 'z':
        return majsoul_to_mjai_dict.get(tile, tile)
    
    # 普通牌直接返回
    return tile


def mjai2majsoul(tileMjai: str) -> str:
    """
    将 mjai 的牌表示转换为雀魂格式
    支持普通牌、字牌和红宝牌
    """
    # 字牌转换字典
    mjai_to_majsoul_dict = {
        'E': '1z',  # 东风
        'S': '2z',  # 南风
        'W': '3z',  # 西风
        'N': '4z',  # 北风
        'P': '5z',  # 白板
        'F': '6z',  # 发财
        'C': '7z'   # 中
    }
    
    # 红宝牌转换
    if tileMjai[-1] == 'r':
        return '0' + tileMjai[1]
    
    # 字牌转换
    if tileMjai in mjai_to_majsoul_dict:
        return mjai_to_majsoul_dict[tileMjai]
    
    # 普通牌直接返回
    return tileMjai


class CardRecorder:
    # 由于雀魂不区分相同牌的编号，但天凤区分tile136，需要根据出现的顺序转换
    # 增加mjai转换？ 但是好像mjai就是雀魂格式
    def __init__(self):
        self.clear()

    def clear(self):
        self.cardDict = {tile: 0 for tile in sdk.all_tiles}

    def majsoul2tenhou(self, tile: str) -> Tuple[int, int]:
        # tileStr to (tile136,tile34) (e.g. '0s' -> (88,37)
        t = 'mpsz'.index(tile[-1])
        if tile[0] == '0':
            #红宝牌
            return [(16, 35), (52, 36), (88, 37)][t]
        else:
            tile136 = (ord(tile[0])-ord('0')-1)*4+9*4*t
            if tile[0] == '5' and t < 3:  # 5 m|p|s
                tile136 += 1
            tile136 += self.cardDict[tile]
            self.cardDict[tile] += 1
            assert(0 <= self.cardDict[tile] <= 4)
            tile34 = tile136//4
            return (tile136, tile34)

    def tenhou2majsoul(self, tile136=None, tile34=None):
        # (tile136,tile34) to tileStr
        if tile136 != None:
            assert(tile34 == None)
            tile136 = int(tile136)
            if tile136 in (16, 52, 88):
                #红宝牌
                return '0'+'mps'[(16, 52, 88).index(tile136)]
            else:
                return str((tile136//4) % 9+1)+'mpsz'[tile136//36]
        else:
            assert(tile136 == None)
            tile34 = int(tile34)
            if tile34 > 34:
                #红宝牌
                return '0'+'mps'[tile34-35]
            else:
                return str(tile34 % 9+1)+'mpsz'[tile34//9]
            


class AIWrapper(sdk.GUIInterface, sdk.MajsoulHandler):
    # TenHouAI <-> AI_Wrapper <-> Majsoul Interface

    def __init__(self):
        super().__init__()
        self.AI_socket = None
        # 与Majsoul的通信
        self.majsoul_server = ServerProxy(
            "http://127.0.0.1:37247")   # 初始化RPC服务器
        self.liqiProto = sdk.LiqiProto()
        # 牌号转换
        self.cardRecorder = CardRecorder()
        # 添加日志文件句柄
        self.log_file = open('mjai_communication.log', 'a', encoding='utf-8')

    def init(self, socket_: socket.socket):
        # 设置与AI的socket链接并初始化
        self.AI_socket = socket_
        self.AI_buffer = bytes(0)
        self.AI_state = State.Playing # 只支持Remote模式
        #  与Majsoul的通信
        self.majsoul_history_msg = []   # websocket flow_msg
        self.majsoul_msg_p = 0  # 当前准备解析的消息下标
        self.liqiProto.init()
        # AI上一次input操作的msg_dict(维护tile136一致性)
        #self.lastOp = self.tenhouEncode({'opcode': None})
        self.lastOp = self.mjaiEncode({'type': None})
        self.lastDiscard = None         # 牌桌最后一次出牌tileMjai，用于吃碰杠牌号
        self.hai = []                   # 我当前手牌的tileMjai编号(和AI一致)
        self.isLiqi = False             # 当前是否处于立直状态
        self.wait_a_moment = False      # 下次操作是否需要额外等待
        self.lastSendTime = time.time()  # 防止操作过快
        self.pengInfo = dict()          # 记录当前碰的信息，以维护加杠时的一致性
        self.lastOperation = None       # 用于判断吃碰是否需要二次选择

    def isPlaying(self) -> bool:
        # 从majsoul websocket中获取数据，并判断数据流是否为对局中
        n = self.majsoul_server.get_len()
        liqiProto = sdk.LiqiProto()
        if n == 0:
            return False
        try:
            flow = pickle.loads(self.majsoul_server.get_items(0, min(100, n)).data)
            for flow_msg in flow:
                try:
                    result = liqiProto.parse(flow_msg)
                    if result.get('method', '') == '.lq.FastTest.authGame':
                        return True
                except KeyError as e:
                    # 忽略未知的协议方法
                    print(f"Warning: Unknown protocol method: {e}")
                    continue
                except Exception as e:
                    print(f"Error parsing message: {e}")
                    continue
        except Exception as e:
            print(f"Error processing websocket data: {e}")
            return False
        return False

    def recvFromMajsoul(self):
        # 从majsoul websocket中获取数据，并尝试解析执行。
        # 如果未达到要求无法执行则锁定self.majsoul_msg_p直到下一次尝试。
        n = self.majsoul_server.get_len()
        l = len(self.majsoul_history_msg)
        if l < n:
            flow = pickle.loads(self.majsoul_server.get_items(l, n).data)
            self.majsoul_history_msg = self.majsoul_history_msg+flow
            pickle.dump(self.majsoul_history_msg, open(
                'websocket_frames.pkl', 'wb'))
        if self.majsoul_msg_p < n:
            flow_msg = self.majsoul_history_msg[self.majsoul_msg_p]
            result = self.liqiProto.parse(flow_msg)
            #print(result)
            failed = self.parse(result)
            if not failed:
                self.majsoul_msg_p += 1

    def recv(self, data: bytes):
        #接受来自AI的Mjai数据
        self.AI_buffer += data
        
        s = self.AI_buffer.split(b'\n')
        for msg in s[:-1]:
            decoded_msg = msg.decode('utf-8')
            # 记录接收到的消息
            self.log_file.write(f'RECV: {decoded_msg}\n')
            self.log_file.flush()  # 立即写入文件
            print('recv:', decoded_msg)
            self._eventHandler(decoded_msg)
        self.AI_buffer = s[-1]

    def send(self, data: Dict):
        # 向AI发送mjai格式数据
        msg = self.mjaiEncode(data)
        if msg:
            # 记录发送的消息
            self.log_file.write(f'SEND: {msg}\n')
            self.log_file.flush()  # 立即写入文件
            print('send:', msg)
            self.AI_socket.send(msg.encode()+b'\n')
            self.lastSendTime = time.time()
                
    def _eventHandler(self, msg):
        # 解析AI发来的数据
        d = self.mjaiDecode(msg)
        if not d:
            return  # 解码失败，直接返回

        if self.AI_state == State.WaitingForStart:
            funcName = 'on_' + d['type']
            if hasattr(self, funcName):
                getattr(self, funcName)(d)
            else:
                print('[AI EVENT] :', msg)
        elif self.AI_state == State.Playing:
            op = d['type']
            if op == 'dahai':
                # 出牌
                self.on_DiscardTile(d)
            elif op == 'reach':
                # 宣告自己立直
                self.on_Liqi()
                # 请求弃牌方案
                time.sleep(2)
                self.send({'type': 'reach', 'actor': self.mySeat})
            else:
                # 回应吃碰杠
                self.on_ChiPengGang(d)
    
    def mjaiDecode(self, msg: str) -> Dict:
        try:
            return json.loads(msg)
        except json.JSONDecodeError as e:
            print(f"Failed to decode mjai message: {e}")
            return {}

    def mjaiEncode(self, kwargs: Dict) -> str:
        try:
            return json.dumps(kwargs)
        except (TypeError, ValueError) as e:
            print(f"Failed to encode mjai message: {e}")
            return ""

    #-------------------------AI回调函数-------------------------

    def on_NEXTREADY(self, msg_dict):
        # newRound
        self.AI_state = State.Playing
    #-------------------------Majsoul回调函数-------------------------
    
    def authGame(self, accountId: int, seatList: List[int]):
        """
        accountId:我的userID
        seatList:所有人的userID(从东家开始顺序)
        """
        assert(len(seatList) == 4)
        
        for i, seat in enumerate(seatList):
            if seat == accountId:
                self.mySeat = i
                break
            
        # 构建 mjai 的开始游戏消息
        msg_dict = {
            'type': 'start_game',
            'kyoku_first': 0, # kyoku_first: 4 -> 东, 0 -> 南
            'id': self.mySeat,
            'names': ['a','b','c','d']
        }
        
        # 发送开始游戏的消息
        self.send(msg_dict)

    def newRound(self, chang: int, ju: int, ben: int, liqibang: int, tiles: List[str], scores: List[int], leftTileCount: int, doras: List[str]):
        """
        chang: 当前的场风，0~3: 东南西北
        ju: 当前第几局 (0:1局,3:4局，连庄不变)
        liqibang: 流局立直棒数量 (画面左上角一个红点的棒)
        ben: 连装棒数量 (画面左上角八个黑点的棒)
        tiles: 我的初始手牌
        scores: 当前场上四个玩家的剩余分数 (从东家开始顺序)
        leftTileCount: 剩余牌数
        doras: 宝牌列表
        """    
        assert chang * 4 + ju >= 0
        assert len(tiles) in (13, 14) and all(tile in sdk.all_tiles for tile in tiles)
        assert leftTileCount == 69
        assert all(dora in sdk.all_tiles for dora in doras)
        assert len(doras) == 1

        if self.AI_state != State.Playing:
            return True  # AI未准备就绪，停止解析

        # 初始化状态
        self.isLiqi = False
        self.cardRecorder.clear()
        self.pengInfo.clear()


        # 设置玩家手牌
        self.scores = scores
        self.hai = tiles[:13]
        oya = ju % 4  # 当前庄家，按照局数 (从东家开始)
        
        tehais = [["?","?","?","?","?","?","?","?","?","?","?","?","?"], 
                  ["?","?","?","?","?","?","?","?","?","?","?","?","?"], 
                  ["?","?","?","?","?","?","?","?","?","?","?","?","?"], 
                  ["?","?","?","?","?","?","?","?","?","?","?","?","?"]]        
        tehais[self.mySeat] = tiles[:13] 
        
        # 构建 mjai 的开始牌局消息
        msg_dict = {
            'type': 'start_kyoku',
            'bakaze': "ESWN"[chang],  # 场风
            'kyoku': ju + 1,  # 局数 (1-indexed)
            'honba': ben,     # 连装棒数量
            'kyotaku': liqibang,  # 流局立直棒数量
            'dora_marker': doras[0], # 宝牌指示牌
            'scores': scores,  # 初始分数
            'oya': oya,  # 庄家
            'tehais': tehais  # 手牌
        }

        # 发送开始游戏的消息
        self.send(msg_dict)

        # 如果玩家是庄家，等待第一轮
        #if oya == self.mySeat:
        self.wait_a_moment = True

        # 如果玩家摸牌 (14 张手牌情况)
        if len(tiles) == 14:
            self.iDealTile(self.mySeat, tiles[13], leftTileCount, {}, {})

    def newDora(self, dora: str):
        """
        处理discardTile/dealTile中通知增加明宝牌的信息
        """
        #tile136, _ = self.cardRecorder.majsoul2tenhou(dora)
        #self.send(self.tenhouEncode({'opcode': 'DORA', 'hai': tile136}))
        self.send({'type': 'dora', 'dora_marker': dora})

    def discardTile(self, seat: int, tile: str, moqie: bool, isLiqi: bool, operation):
        """
        seat:打牌的玩家
        tile:打出的手牌
        moqie:是否是摸切
        isLiqi:当回合是否出牌后立直
        operation:可选动作(吃碰杠)
        """
        assert(0 <= seat < 4)
        assert(tile in sdk.all_tiles)
        if isLiqi:
            if self.mySeat == seat:
                self.isLiqi = True
            else:
                msg_dict = {'type': 'reach', 'actor': seat}
                self.send(msg_dict)
            
        
        actions = []
        # 处理可能的后续操作
        if operation is not None:
            assert(operation.get('seat', 0) == self.mySeat)
            opList = operation.get('operationList', [])
            
            # 记录可能的动作
            if any(op['type'] == Operation.Chi.value for op in opList):
                actions.append('chi')
            if any(op['type'] == Operation.Peng.value for op in opList):
                actions.append('pon')
            if any(op['type'] == Operation.MingGang.value for op in opList):
                actions.append('kan')
            if any(op['type'] == Operation.Hu.value for op in opList):
                actions.append('hora')
                
        msg_dict = {
            'type': 'dahai',
            'actor': seat,
            'pai': tile,
            'tsumogiri': moqie , # True 表示摸切，False 表示手切
            'possible_actions' : actions # test 
        }
        self.send(msg_dict)
    
        # 更新内部状态
        self.lastDiscard = tile
        self.lastDiscardSeat = seat
        self.lastOperation = operation

    def dealTile(self, seat: int, leftTileCount: int, liqi: Dict):
        """
        seat: 摸牌的玩家
        leftTileCount: 剩余牌数
        liqi: 如果上一回合玩家出牌立直，则紧接着的摸牌阶段有此参数表示立直成功
        """
        assert 0 <= seat < 4
        assert isinstance(liqi, dict) or liqi is None

        # 处理立直成功
        if liqi:
            actor = liqi.get('seat', 0)
            score = liqi.get('score', 0)
            # 更新玩家分数
            self.scores[actor] = score
            # 发送立直成功的 mjai 消息
            msg_dict = {'type': 'reach_accepted', 'actor': actor , 'pai': "?"}
            self.send(msg_dict)

        # 发送摸牌消息
        msg_dict = {'type': 'tsumo', 'actor': seat, 'pai': "?"}
        self.send(msg_dict)


    def iDealTile(self, seat: int, tile: str, leftTileCount: int, liqi: Dict, operation: Dict):
        """
        seat: 我自己
        tile: 摸到的牌
        leftTileCount: 剩余牌数
        liqi: 如果上一回合玩家出牌立直，则紧接着的摸牌阶段有此参数表示立直成功
        operation: 可选操作列表
        """
        assert seat == self.mySeat
        assert tile in sdk.all_tiles
        assert isinstance(liqi, dict) or liqi is None

        # 处理立直成功
        if liqi:
            actor = liqi.get('seat', 0)
            score = liqi.get('score', 0)
            # 更新玩家分数
            self.scores[actor] = score
            # 发送立直成功的 mjai 消息
            msg_dict = {'type': 'reach_accepted', 'actor': actor}
            self.send(msg_dict)

        # 记录摸到的牌
        self.hai.append(tile)

        # 发送摸牌消息
        msg_dict = {'type': 'tsumo', 'actor': seat, 'pai': tile}
        self.send(msg_dict)

        # 处理可选操作
        if operation is not None:
            opList = operation.get('operationList', [])
            for op in opList:
                op_type = op['type']
                if op_type == Operation.Zimo.value or op_type == Operation.Hu.value:
                    # 自摸和牌
                    #msg_dict = {'type': 'hora', 'actor': seat, 'pai': tile , 'target': seat} # mortal要求自摸target为自己
                    #self.send(msg_dict)
                    
                    self.actionZimo()
                    # 你妈叫你点自摸按钮拉
                    
                    pass
                elif op_type == Operation.Liqi.value:
                    # 立直
                    #msg_dict = {'type': 'reach', 'actor': seat}
                    #self.send(msg_dict)
                    pass
                elif op_type == Operation.JiaGang.value:
                    # 加杠
                    #msg_dict = {'type': 'kan', 'actor': seat, 'pai': tile}
                    #self.send(msg_dict)
                    pass

    def chiPengGang(self, type_: int, seat: int, tiles: List[str], froms: List[int], tileStates: List[int]):
        """
        type_: 操作类型
        seat: 执行吃、碰、杠的玩家
        tiles: 吃、碰、杠的牌组
        froms: 每张牌的来源玩家
        tileStates: 未知（TODO）
        """
        assert 0 <= seat < 4
        assert all(tile in sdk.all_tiles for tile in tiles)
        assert all(0 <= i < 4 for i in froms)
        assert seat != froms[-1]

        # 获取最后打出的牌，已为 mjai 格式
        last_discard = self.lastDiscard
        assert tiles[-1] == last_discard

        # 动作发起者和目标玩家
        actor = seat
        target = froms[-1]

        # 构建 mjai 消息字典
        if type_ == 0:
            # 吃
            assert len(tiles) == 3
            msg_dict = {
                'type': 'chi',
                'actor': actor,
                'target': target,
                'pai': last_discard,
                'consumed': tiles[:-1]
            }
        elif type_ == 1:
            # 碰
            assert len(tiles) == 3
            msg_dict = {
                'type': 'pon',
                'actor': actor,
                'target': target,
                'pai': last_discard,
                'consumed': tiles[:-1]
            }
        elif type_ == 2:
            # 明杠
            assert len(tiles) == 4
            msg_dict = {
                'type': 'daiminkan',
                'actor': actor,
                'target': target,
                'pai': last_discard,
                'consumed': tiles[:-1]
            }
        else:
            #raise NotImplementedError("未实现的操作类型")
            pass

        # 发送 mjai 消息
        self.send(msg_dict)


    def anGangAddGang(self, type_: int, seat: int, tiles: str):
        """
        type_: 操作类型
        seat: 执行杠的玩家
        tiles: 杠的牌
        """
        assert 0 <= seat < 4

        # 动作发起者
        actor = seat
        
        
        def popHai(tile):
            #从self.hai中找到tile并pop
            for mtile in self.hai:
                if mtile == tile:
                    self.hai.remove(mtile)
                    return mtile
            raise Exception(tile+' not found.')

        # 构建 mjai 消息字典
        if type_ == 2:
            # 加杠（加杠在 mjai 中使用 'kakan' 表示）
            tiless = [tiles]*3
            if tiles[0] == '5':
                if tiles[-1] != 'r':
                    tiless[-1] = '5'+tiles[-1]+'r'
            msg_dict = {
                'type': 'kakan',
                'actor': actor,
                'pai': tiles ,
                'consumed': tiless
            }
            if seat == self.mySeat:
                popHai(tiles)
            else:
                pass
        elif type_ == 3:
            # 暗杠（暗杠在 mjai 中使用 'ankan' 表示）
            
            if tiles[0] == '5' and tiles[-1] == 'r':
                #红宝牌
                tiles = '5'+tiles[1]
                
            tiless = [tiles]*4
            
            if tiles[0] == '5':
                tiless[-1] = '5'+tiles[-1]+'r'
            
            msg_dict = {
                'type': 'ankan',
                'actor': actor,
                'consumed': tiless
            }
            if seat == self.mySeat:
                if tiles[0] == '5':
                    for i in range(3):
                        popHai(tiles)
                    popHai('5'+tiles[-1]+'r')
                else :
                    for i in range(4):
                        popHai(tiles)
            else :
                pass
        else:
            #raise NotImplementedError("未实现的操作类型")
            pass

        # 发送 mjai 消息
        self.send(msg_dict)


    def hule(self, hand: List[str], huTile: str, seat: int, zimo: bool, liqi: bool, doras: List[str], liDoras: List[str], fan: int, fu: int, oldScores: List[int], deltaScores: List[int], newScores: List[int]):
        """
        hand: 胡牌者手牌
        huTile: 和牌的牌
        seat: 玩家座次
        zimo: 是否自摸
        liqi: 是否立直
        doras: 明宝牌列表
        liDoras: 里宝牌列表
        fan: 番数
        fu: 符数
        oldScores: 4人旧分
        deltaScores: 新分减旧分
        newScores: 4人新分
        """
        assert all(tile in sdk.all_tiles for tile in hand)
        assert huTile in sdk.all_tiles
        assert 0 <= seat < 4
        assert all(tile in sdk.all_tiles for tile in doras)
        assert all(tile in sdk.all_tiles for tile in liDoras)

        # 动作发起者
        actor = seat

        # 构建 mjai 消息字典
        msg_dict = {
            'type': 'hora',
            'actor': actor,
            'target': actor if zimo else self.lastDiscardSeat,
            'pai': huTile,
            'hora_tehais': hand,
            'dora_marker': doras,
            'uradora_marker': liDoras,
            'fan': fan,
            'fu': fu,
            'ten': deltaScores[seat],
            'old_scores': oldScores,
            'new_scores': newScores
        }

        # 发送 mjai 消息
        self.send(msg_dict)
        #self.AI_state = State.WaitingForStart 

    def liuju(self, tingpai: List[bool], hands: List[List[str]], oldScores: List[int], deltaScores: List[int]):
        """
        tingpai: 4个玩家是否听牌
        hands: 听牌玩家的手牌（未听牌为 []）
        oldScores: 4人旧分
        deltaScores: 新分减旧分
        """
        assert all(tile in sdk.all_tiles for hand in hands for tile in hand)

        # 构建 mjai 消息字典
        msg_dict = {
            'type': 'ryukyoku',
            'reason': 'nm'  # 荒牌流局
        }

        # 添加各玩家的听牌状态和手牌
        tenpai_info = []
        for i in range(4):
            player_info = {
                'type': 'tenpai' if tingpai[i] else 'noten',
                'pai': hands[i] if tingpai[i] else []
            }
            tenpai_info.append(player_info)
        msg_dict['tenpai'] = tenpai_info

        # 计算新分数
        newScores = [old + delta for old, delta in zip(oldScores, deltaScores)]

        # 添加分数信息
        msg_dict['old_scores'] = oldScores
        msg_dict['new_scores'] = newScores

        # 发送 mjai 消息
        self.send(msg_dict)

        # self.AI_state = State.WaitingForStart

    def specialLiuju(self):
        """
        处理特殊流局情况，包括四风连打、九种九牌、四杠散了。
        """
        # 假设特殊流局没有明确标注，可以根据实际逻辑选择默认的流局原因
        msg_dict = {
            'type': 'ryukyoku',
            'reason': 'four_winds'  # 默认设置为四风连打，或者根据实际逻辑动态调整
        }
        
        # 发送 mjai 消息
        self.send(msg_dict)


    #-------------------------Majsoul动作函数-------------------------

    def wait_for_a_while(self, delay=1):
        # 如果读秒不足delay则强行等待一会儿
        dt = time.time()-self.lastSendTime
        if dt < delay:
            time.sleep(delay-dt)

    def get_exponential_delay(self):
        """
        生成1-5秒的随机延迟，使用指数分布
        5秒的概率在1%-5%之间，1秒的概率最大
        """
        # 设置lambda参数来控制分布
        lambda_param = 1.1
        
        while True:
            # 生成指数分布随机数
            x = np.random.exponential(scale=1/lambda_param)
            
            # 将值映射到1-5的范围
            delay = 0.5 + x
            
            # 如果延迟在1-5之间，则返回
            if 0.5 <= delay <= 5:
                return delay

    def on_DiscardTile(self, msg_dict):
        if self.wait_a_moment == True:
            self.wait_a_moment = False
            time.sleep(2)
        self.lastOp = msg_dict
        tile = mjai2majsoul(msg_dict["pai"])
        if not self.isLiqi:
            self.wait_for_a_while(delay=0.5)
            self.forceTiaoGuo()
            self.wait_for_a_while(delay=self.get_exponential_delay())
            self.actionDiscardTile(tile)

    def on_ChiPengGang(self, msg_dict):
        # <N ...\>
        self.wait_for_a_while(delay=0.5)
        if ('type' not in msg_dict) or (msg_dict['type'] == "none"):
            # 检查是否有和牌操作
            if self.lastOperation is not None:
                opList = self.lastOperation.get('operationList', [])
                if any(op['type'] == Operation.Hu.value for op in opList):
                    self.checkZimo()
                    
                # 检查是否有吃碰杠操作
                has_chi_pon_kan = any(op['type'] in [Operation.Chi.value, Operation.Peng.value, Operation.MingGang.value] 
                                    for op in opList)
                
                # 检查meta和q_values条件
                if (has_chi_pon_kan and 
                    'meta' in msg_dict and 
                    'q_values' in msg_dict['meta'] and 
                    isinstance(msg_dict['meta']['q_values'], list) and 
                    len(msg_dict['meta']['q_values']) > 0 and 
                    msg_dict['meta']['q_values'][-1] == max(msg_dict['meta']['q_values'])):
                    
                    self.actionChiPengGang(sdk.Operation.NoEffect, [])
                
            return
        
        self.wait_for_a_while(delay=2)
        
        type_ = msg_dict['type']
        
        # 获取用于副露的牌列表
        consumed = msg_dict.get('consumed', [])
        
        # 将牌从 mjai 格式转换为程序内部格式
        consumed_tiles = [mjai2majsoul(tile) for tile in consumed]
        
        if type_ == "pon":
            #碰        
            self.actionChiPengGang(sdk.Operation.Peng, consumed_tiles)
        elif type_ == "daiminkan":
            #明杠
            self.actionChiPengGang(sdk.Operation.MingGang, consumed_tiles)
        elif type_ == "chi":
            #吃        
            self.actionChiPengGang(sdk.Operation.Chi, consumed_tiles)
            #判断是否有多个候选方案需二次选择
            if self.lastOperation != None:
                opList = self.lastOperation.get('operationList', [])
                opList = [op for op in opList if op['type']
                          == Operation.Chi.value]
                assert(len(opList) == 1)
                op = opList[0]
                combination = op['combination']
                # e.g. combination = ['4s|0s', '4s|5s']
                if len(combination) > 1:
                    # 需要二次选择
                    combination = [tuple(sorted(c.split('|')))
                                   for c in combination]
                    AI_combination = tuple(sorted(consumed_tiles))
                    assert(AI_combination in combination)
                    # 如果有包含红包牌的同构吃但AI犯蠢没选，强制改为吃红包牌
                    oc = tuple(sorted([i.replace('5', '0')
                                       for i in AI_combination]))
                    if oc in combination:
                        AI_combination = oc
                    print('clickCandidateMeld AI_combination', AI_combination)
                    time.sleep(2)
                    self.clickCandidateMeld(AI_combination)
        elif type_ == "ankan":
            # 暗杠
            self.actionChiPengGang(sdk.Operation.MingGang, consumed_tiles)
        elif type_ == "kakan":
            # 加杠
            self.actionChiPengGang(sdk.Operation.JiaGang, consumed_tiles)
        elif type_ == "hora" and msg_dict.get('actor', -1) != self.mySeat:
            self.actionHu(similarityThreshold = 0.48)
        elif type_ == "hora" and msg_dict.get('actor', -1) == self.mySeat:
            self.actionZimo()   
        elif type_ == "ryukyoku":
            # 流局
            self.actionRyukyoku()
        else:
            #raise NotImplementedError(f"未实现的操作类型: {type_}")
            pass

    def on_Liqi(self):
        self.wait_for_a_while()
        self.actionLiqi()

    def __del__(self):
        # 确保文件被正确关闭
        if hasattr(self, 'log_file'):
            self.log_file.close()


def MainLoop(isRemoteMode=False, remoteIP: str = None, level=None, duration=None):
    
    start_time = time.time()
    # 循环进行段位场对局，level=0~4表示铜/银/金/玉/王之间，None需手动开始游戏
    # calibrate browser position
    aiWrapper = AIWrapper()
    print('waiting to calibrate the browser location')
    while not aiWrapper.calibrateMenu():
        print('  majsoul menu not found, calibrate again')
        time.sleep(3)

    while True:
        # create AI
        if isRemoteMode == False:
            print('create AI subprocess locally')
            AI = Popen('python main.py --fake', cwd='JianYangAI',
                       creationflags=CREATE_NEW_CONSOLE)
            # create server
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_address = ('127.0.0.1', 7479)
            print('starting up on %s port %s' % server_address)
            server.bind(server_address)
            server.listen(1)
            print('waiting for the AI')
            connection, client_address = server.accept()
            print('AI connection: ', type(connection),
                  connection, client_address)

        else:
            print('call remote AI')
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port = importlib.import_module('remote').REMOTE_PORT
            connection.connect((remoteIP, port))
            ACK = connection.recv(3)
            assert(ACK == b'ACK')
            print('remote AI connection: ', connection, remoteIP)

        aiWrapper.init(connection)
        inputs = [connection]
        outputs = []

        if (level != None and 
            (duration is None or time.time() - start_time + 40*60 < duration*60) and 
            os.path.exists('canloop.flag')):
            aiWrapper.actionBeginGame(level)

        print('waiting for the game to start')
        while not aiWrapper.isPlaying():
            time.sleep(3)

        while True:
            readable, writable, exceptional = select.select(
                inputs, outputs, inputs, 0.1)
            for s in readable:
                data = s.recv(1024)
                if data:
                    # A readable client socket has data
                    aiWrapper.recv(data)
                else:
                    # Interpret empty result as closed connection
                    print('closing server after reading no data')
                    return
            # Handle "exceptional conditions"
            for s in exceptional:
                print('handling exceptional condition for', s.getpeername())
                break
            aiWrapper.recvFromMajsoul()
            if aiWrapper.isEnd:
                results = [rv for r in zip(aiWrapper.finalScore, [-1]*4) for rv in r]
                aiWrapper.send('owari="{},{},{},{},{},{},{},{}"\x00<PROF\x00'.format(*results))
                aiWrapper.isEnd = False
                connection.close()
                if isRemoteMode == False:
                    AI.wait()
                aiWrapper.actionReturnToMenu()
                break


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="MajsoulAI")
    parser.add_argument('-r', '--remote_ip', default='')
    parser.add_argument('-l', '--level', default=None)
    parser.add_argument('-d', '--duration', type=int, default=None)
    parser.add_argument('--test', action='store_true', help='Force all delays to 0.5s')  # 添加测试参数
    args = parser.parse_args()
    level = None if args.level == None else int(args.level)
    duration = args.duration

    # 如果是测试模式，修改 wait_for_a_while 函数
    if args.test:
        def new_wait_for_a_while(self, delay=1):
            dt = time.time()-self.lastSendTime
            delay = 0.2
            if dt < delay:  # 强制使用 0.5s 延迟
                time.sleep(delay-dt)
        AIWrapper.wait_for_a_while = new_wait_for_a_while

    if level != None:
        with open('canloop.flag', 'w') as f:
            pass  # 不写入任何内容
    if args.remote_ip == '':
        MainLoop(level=level, duration=duration)
    else:
        MainLoop(isRemoteMode=True, remoteIP=args.remote_ip, level=level, duration=duration)
