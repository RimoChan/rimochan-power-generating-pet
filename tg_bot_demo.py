import json
import time
import logging
import threading
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

import requests

from 莉沫酱 import 莉沫酱


灵牌 = 'xxxx'
api_base = f'https://api.telegram.org/bot{灵牌}/'
my_chat_id = 12345678


@retry(stop=stop_after_attempt(50), wait=wait_fixed(10), reraise=True)
def _requests_get(*args, **kwargs):
    resp = requests.get(*args, **kwargs)
    if resp.status_code != 200:
        raise Exception(resp.content)
    return resp


def g():
    offset = 0
    if Path('offset.txt').is_file():
        with open('offset.txt') as f:
            offset = int(f.read())
    resp = _requests_get(api_base + 'getUpdates', params={'offset': offset})
    data = resp.json()
    if not data['ok']:
        logging.warning('坏耶')
    r = data['result']
    if r:
        print(r)
        offset = max([u['update_id'] for u in r]) + 1
        with open('offset.txt', 'w') as f:
            f.write(str(offset))
    for u in r:
        手(u)


def 送(事件, a):
    if 事件 in ['被动说话', '主动说话']:
        _requests_get(api_base + 'sendMessage', params={'chat_id': my_chat_id, 'text': a})
    elif 事件 in ['事件']:
        _requests_get(api_base + 'sendMessage', params={'chat_id': my_chat_id, 'text': f'【{a[1]}】'})


记忆 = None
if Path('记忆.json').is_file():
    记忆 = json.load(open('记忆.json', 'r', encoding='utf8'))
莉 = 莉沫酱(输出=送, 记忆=记忆)
threading.Thread(target=莉.启动, daemon=True).start()


def 手(u: dict):
    if u.get('message', {}).get('chat', {}).get('id') != my_chat_id:
        return
    text = u['message']['text']
    莉.主动反应(text)


while True:
    g()
    s = json.dumps(莉.记忆, ensure_ascii=False, indent=2)
    with open('记忆.json', 'w', encoding='utf8') as f:
        f.write(s)
    time.sleep(1)
