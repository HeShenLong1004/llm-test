import pandas as pd
import requests
import time
from tqdm import tqdm
# Flask服务的URL
url = 'http://localhost:10000/llm_intent'

data = {'dn': 'C411E044B84A', 'log_id': 'sdlc-0187dca413e44b289b79470825dd7a66', 'seq_id': 1, 'asr_text': '窗帘关三分之一', 'ip_address': '39.69.107.98', 'city': '山东省潍坊市高密市月潭路', 'open_migu': False}
start = time.time()
# 发送POST请求
response = requests.post(url, json=data)
end = time.time()
print(f'调用耗时：{end-start}')
print(response.text)

# import requests
# import pandas as pd
# # 临时设置显示所有行和列
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# # 定义URL
# url = 'https://prod-auth.iotbull.com/auth/oauth2/token'

# # 定义请求头
# headers = {
#     'Authorization': 'Basic cGFhc2Nsb3VkY2xpZW50dWljYWlsYWI6JDJhJDEwJDZmU0VQN2VXRUd1RVovei8yaC45bE9ndFNOc2FVcnF0VU9ISjlOSTJMTHpncmVyL1JBb1Fh',
#     'Content-Type': 'application/x-www-form-urlencoded'
# }

# # 定义请求数据
# data = {
#     'scope': 'server',
#     'grant_type': 'client_credentials'
# }

# # 发送POST请求
# response = requests.post(url, headers=headers, data=data)
# print(response.json())
# # 打印响应内容
# access_token = response.json()['access_token']


# #C411E1018DB6
# 定义Mock地址
# mock_url = f"https://skill.iotbull.com/skill/ai/voice/device/list?deviceName=C411E0451F5E"

# # 定义请求头
# headers = {
#     'Content-Type': 'application/json',
#     # 如果需要认证，请添加Authorization头
#     'Authorization': 'Bearer '+access_token  # 替换为你的实际服务鉴权token
# }

# # 发送GET请求
# response = requests.get(mock_url, headers=headers)

# # 检查响应状态码
# if response.status_code == 200:
#     print("请求成功，响应内容：")
#     print(response.text)  # 打印JSON响应内容
#     # print(pd.DataFrame(response.json()['result']).columns)
#     print(pd.DataFrame(response.json()['result']))
# else:
#     print("请求失败，状态码：", response.status_code)
#     print("响应内容：", response.text)


# import dashscope
# from http import HTTPStatus
# from config import api_key, reply_prompt, device_type
# # 如果环境变量配置无效请启用以下代码
# dashscope.api_key = api_key
# import numpy as np
# from typing import Generator, List

# # 最多支持25条，每条最长支持2048tokens
# DASHSCOPE_MAX_BATCH_SIZE = 25

# def batched(inputs: List,
#             batch_size: int = DASHSCOPE_MAX_BATCH_SIZE) -> Generator[List, None, None]:
#     for i in range(0, len(inputs), batch_size):
#         yield inputs[i:i + batch_size]

# def embed_with_list_of_str(inputs: List):
#     result = None  # merge the results.
#     batch_counter = 0
#     for batch in batched(inputs):
#         resp = dashscope.TextEmbedding.call(
#             model=dashscope.TextEmbedding.Models.text_embedding_v2,
#             text_type='query',
#             input=batch)
#         if resp.status_code == HTTPStatus.OK:
#             if result is None:
#                 result = resp
#             else:
#                 for emb in resp.output['embeddings']:
#                     emb['text_index'] += batch_counter
#                     result.output['embeddings'].append(emb)
#                 result.usage['total_tokens'] += resp.usage['total_tokens']
#         else:
#             print(resp)
#         batch_counter += len(batch)
#     return result

# def cosine_similarity(vector_a, vector_b):
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)
#     if norm_a == 0 or norm_b == 0:
#         return 0  # 避免除以零的情况
#     return dot_product / (norm_a * norm_b)

# if __name__ == '__main__':
#     inputs =  ['执行茶室关闭茶室灯光','关灯']
#     result = embed_with_list_of_str(inputs)
#     embeddings = result['output']['embeddings']
#     embedding_v1 = embeddings[0]['embedding']
#     embedding_v2 = embeddings[1]['embedding']
#     print(inputs)
#     print(cosine_similarity(np.array(embedding_v1),np.array(embedding_v2)))
#     print('*'*100)





# import os
# import pandas as pd
# import json
# import time
# from device_control import *
# from intent_recognition import intent_recognition
# from flask import Flask, request, jsonify
# import uuid
# from datetime import datetime
# from config import skill_name, command_name, setup_logger, llm_model_name, question_scene
# import requests
# import redis
# import random
# import concurrent.futures
# from qqwry import QQwry
# import logging
# import numpy as np
# from tqdm import tqdm

# r = redis.Redis(
#     host='r-uf6lg8sdnxfeyu3n4k.redis.rds.aliyuncs.com',  # 阿里云 Redis 实例地址
#     port=6379,                                           # Redis 端口
#     password='Gnailab2024aaa333',                        # Redis 密码
#     db=0,                                                # 使用的数据库编号
#     decode_responses=False                                 # 如果需要返回字符串而不是字节，可以设置这个选项为 True
# )

# logger = setup_logger('llm serving','serving.log')

# def get_embedding_from_redis(key):
#     # Build the full Redis key
#     redis_key = 'embedding_v1:' + key
#     # Fetch embedding from Redis; returns None if the key does not exist
#     embedding = r.get(redis_key)
#     if embedding:
#         # Refresh expiration to one week (604800 seconds)
#         r.expire(redis_key, 604800)
#         return np.frombuffer(embedding, dtype=np.float32)
#     return None

# def batch_get_embeddings_from_redis(keys):
#     """
#     批量从 Redis 获取 embeddings，返回 dict(key->vector) 和缺失 key 列表
#     """
#     pipe = r.pipeline()
#     for key in keys:
#         pipe.get('embedding_v1:' + key)
#     raw_results = pipe.execute()

#     redis_map = {}
#     missing_keys = []
#     existing_keys = []

#     for key, raw in zip(keys, raw_results):
#         if raw is None:
#             missing_keys.append(key)
#         else:
#             vec = np.frombuffer(raw, dtype=np.float32)
#             redis_map[key] = vec
#             existing_keys.append(key)

#     # 刷新过期时间
#     if existing_keys:
#         expire_pipe = r.pipeline()
#         for key in existing_keys:
#             expire_pipe.expire('embedding_v1:' + key, 604800)
#         expire_pipe.execute()

#     return redis_map, missing_keys

# def save_embedding_to_redis(key, embedding):
#     # Save embedding to Redis as bytes and set expiration to one week (604800 seconds)
#     # Using the 'ex' parameter to set expiry time directly
#     r.set('embedding_v1:' + key, embedding.tobytes(), ex=604800)

# def embed_with_list_of_str(text_list):
#     embedding_url = "http://172.18.64.209:6667/embedding"
#     headers = {"Content-Type": "application/json","Authorization":"Bearer 3jK8Lm#9qX2p@Vf7GhT5yN1bR4sW6eD!QzXcVbNmGhJkL"}
#     data = {
#         "texts": text_list
#     }
#     try:
#         logger.info(f'request embedding_url: {embedding_url}')
#         response = requests.post(embedding_url, json=data, headers=headers, timeout=0.5)
#         if response.status_code == 200:
#             return  response.json()
#         else:
#             logger.error(f"请求失败:{response.text}")
#             return None
#     except Exception as e:
#         logger.error("embedding服务请求异常: %s", str(e))
#         return None

# def cosine_similarity(vector_a, vector_b):
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)
#     if norm_a == 0 or norm_b == 0:
#         return 0  # 避免除以零的情况
#     return dot_product / (norm_a * norm_b)

# def scene_instruct(input_text, scene_list, local, unique_id, seq_id, threshold=0.6):
#     state = {}
#     answer_scene_map = {}
#     scene_type_mapping = {}
#     scene_id_to_info = {}

#     # 构建 state, 映射数据结构
#     for info in scene_list:
#         sid = info['sceneId']
#         scene_id_to_info[sid] = info
#         scene_type_mapping[sid] = info.get('sceneType', 1)

#         base = info['sceneName'].replace('场景','').replace('模式','')
#         custom = info['customLightEffectName'].replace('场景','').replace('模式','')
#         suffixes = ['场景','模式','']
#         verbs = ['','打开','执行']
#         rooms = [None]
#         if info.get('roomName'):
#             rooms.append(info['roomName'])

#         for room in rooms:
#             prefix = room or ''
#             for verb in verbs:
#                 for name, ans in [(base, info['sceneName']), (custom, info['customLightEffectName'])]:
#                     for suf in suffixes:
#                         key = f"{verb}{prefix}{name}{suf}"
#                         state[key] = f"{sid}:1"
#                         answer_scene_map[key] = ans

#         # 预设问句匹配
#         for q in question_scene.get(info['sceneName'], []):
#             state[q] = f"{sid}:1"
#             answer_scene_map[q] = info['sceneName']
#         for q in question_scene.get(info['customLightEffectName'], []):
#             state[q] = f"{sid}:1"
#             answer_scene_map[q] = info['customLightEffectName']

#     keys_list = list(state.keys())
#     embeddings = [None] * len(keys_list)

#     # 从 Redis 批量获取
#     redis_map, missing = batch_get_embeddings_from_redis(keys_list)
#     for key, vec in redis_map.items():
#         idx = keys_list.index(key)
#         embeddings[idx] = vec

#     # 批量请求缺失和 input_text
#     to_embed = missing + [input_text]
#     res = embed_with_list_of_str(to_embed)
#     if not res or 'output' not in res or 'embeddings' not in res['output']:
#         return {"code": 500, "error": "Embedding 请求失败"}
#     embs = [np.array(x['embedding'], dtype=np.float32) for x in res['output']['embeddings']]

#     # 存回并填充
#     for i, key in enumerate(missing):
#         save_embedding_to_redis(key, embs[i])
#         idx = keys_list.index(key)
#         embeddings[idx] = embs[i]
#     input_emb = embs[-1]

#     # 计算相似度
#     sims = np.array([cosine_similarity(input_emb, v) for v in embeddings])
#     maxv = np.max(sims)
#     max_idxs = np.where(sims == maxv)[0]

#     # 本地场景优先
#     best_idx = None
#     for idx in max_idxs:
#         sid = state[keys_list[idx]].split(':')[0]
#         if scene_id_to_info[sid].get('roomName', '') == local:
#             best_idx = idx
#             break
#     if best_idx is None and max_idxs.size > 0:
#         best_idx = max_idxs[0]

#     # 构建返回
#     if best_idx is not None and sims[best_idx] >= threshold:
#         key = keys_list[best_idx]
#         sid, val = state[key].split(':')
#         match_name = answer_scene_map.get(key, '未知场景')
#         replies = [f'已为你找到{match_name}', f'正在跑步前去帮您执行{match_name}']
#         logger.info(f'log id:{unique_id}, seq id:{seq_id}. 匹配 {key}, 相似度 {sims[best_idx]}')
#         return {
#             "code": 200,
#             "msg": "success",
#             "data": {
#                 "type": "scene",
#                 "query": input_text,
#                 "ability_pool": {},
#                 "iot_detail": {
#                     "scene_id": sid,
#                     "value": int(val),
#                     "sceneType": scene_type_mapping.get(sid, 1),
#                     "reply": random.choice(replies)
#                 }
#             }
#         }
#     return {}

# for i in range(1000):
#     start = time.time()
#     input_text='我要明亮'
#     unique_id='xxx'
#     seq_id=1
#     local='茶室'
#     scene_list = [{'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '我要品茗', 'sceneId': '6ac4ba512f2b45fc9a42ed0c5a5cebff', 'customLightEffectName': '我要品茗', 'sceneType': 1, 'roomId': 9178238, 'roomName': '茶室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '我要氛围', 'sceneId': 'f460e5a924ed4864ab40e8d66805f545', 'customLightEffectName': '我要氛围', 'sceneType': 1, 'roomId': 9178238, 'roomName': '茶室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '我要关灯', 'sceneId': 'dcb633564d7c465cbd99c80bd816ef5d', 'customLightEffectName': '我要关灯', 'sceneType': 1, 'roomId': 9178238, 'roomName': '茶室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '我要明亮', 'sceneId': '0e298273479f4811b9ed7124b7d0dc47', 'customLightEffectName': '我要明亮', 'sceneType': 1, 'roomId': 9178238, 'roomName': '茶室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '明亮模式', 'sceneId': '9848335a259c449f84db7be1ee99e2db', 'customLightEffectName': '明亮模式', 'sceneType': 1, 'roomId': 9178238, 'roomName': '茶室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '阅读模式', 'sceneId': 'af64c700e9ea402980ee8c3fae028a72', 'customLightEffectName': '阅读模式', 'sceneType': 1, 'roomId': 9161235, 'roomName': '卧室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '卧室氛围', 'sceneId': 'eee701fdade7452aa31cb0ebff014d24', 'customLightEffectName': '卧室氛围', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '微光助眠', 'sceneId': '32ec7758dbed4b009b6c40e8e49c96ae', 'customLightEffectName': '微光助眠', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '睡前刷剧', 'sceneId': '7f50f800b5214ee39f92dfbaec47215c', 'customLightEffectName': '睡前刷剧', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '日常灯光', 'sceneId': '09a2970bfcc3496b83fdb9267f4c8fef', 'customLightEffectName': '日常灯光', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '明亮卧室', 'sceneId': '162231efc53c400f81ef07a9e6236256', 'customLightEffectName': '明亮卧室', 'sceneType': 1, 'roomId': 9161235, 'roomName': '卧室'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '随心', 'sceneId': '1053c20e4a2948ad91869c0a596c8150', 'customLightEffectName': '随心', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '浪漫', 'sceneId': '55216880902f4d8e990658df78b17f0a', 'customLightEffectName': '浪漫', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '闲云野鹤', 'sceneId': 'e2a6a3e94dac479db0102e6245681ab4', 'customLightEffectName': '闲云野鹤', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '氛围用餐', 'sceneId': '1bb4e27d235e437391bbfdc7d1f04250', 'customLightEffectName': '氛围用餐', 'sceneType': 1, 'roomId': 9161229, 'roomName': '默认'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '客厅全关', 'sceneId': '7b9bca014d56465486b25abf1ab9e70c', 'customLightEffectName': '客厅全关', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '回家模式', 'sceneId': '14abfc8ab37f4b8ab98a844bb467acec', 'customLightEffectName': '回家模式', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '独酌', 'sceneId': '5f1bd5605d9846dbb6110d12b1a8d268', 'customLightEffectName': '独酌', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '浪漫西餐', 'sceneId': '6ebde7ebf2e344609ff35ccbcdc84e14', 'customLightEffectName': '浪漫西餐', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '明亮模式', 'sceneId': '6d4a904329214ad28cb3e035686d3717', 'customLightEffectName': '明亮模式', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '观影模式', 'sceneId': 'e903421dfc7141919256d01370d7c237', 'customLightEffectName': '观影模式', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '关闭背景灯', 'sceneId': 'e55039c17d024f11ad1605e28245ef4f', 'customLightEffectName': '关闭背景灯', 'sceneType': 1, 'roomId': 9161229, 'roomName': '默认'}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '治愈模式', 'sceneId': 'ee25e4d1ddf746f6b273606a0a332c39', 'customLightEffectName': '治愈模式', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '中餐合宴', 'sceneId': '072be6107b3940f0b5f371dff743e46f', 'customLightEffectName': '中餐合宴', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '餐厅清洁', 'sceneId': '3fcaa3996c8249f08cea513a960aabd4', 'customLightEffectName': '餐厅清洁', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '会友模式', 'sceneId': '401b8c327f114e4196ccccfe018f17c6', 'customLightEffectName': '会友模式', 'sceneType': 1, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '舒缓治愈', 'sceneId': '8cb7e24a76e54c028dd1e147d2038329', 'customLightEffectName': '舒缓治愈', 'sceneType': 0, 'roomId': None, 'roomName': None}, {'userId': 1021531, 'familyName': '沐光无主灯', 'sceneName': '日暮归家', 'sceneId': '1d8de92ae30a4b67a005e0b701265c66', 'customLightEffectName': '日暮归家', 'sceneType': 0, 'roomId': 9161229, 'roomName': '默认'}]

#     result = scene_instruct(input_text, scene_list, local, unique_id, seq_id, threshold=0.9)
#     end = time.time()
#     if result['code'] == 200:
#         pass
#         # print('成功了')
#         # print(f'模型耗时：{end-start}s')
#     else:
#         print('发生错误')

