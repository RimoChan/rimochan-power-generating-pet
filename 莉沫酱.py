import re
import time
import copy
import json
import random
import logging
from typing import Optional, NoReturn, Any, MutableMapping, Callable

from tenacity import retry, stop_after_attempt, wait_fixed
from rimo_storage.cache import disk_cache
import openai

import tiktoken
enc = tiktoken.encoding_for_model('gpt-3.5-turbo')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


公共prompt = '''
你正在模型试验场中，和其他由人类或大模型扮演的角色进行对话。
你是一个电子宠物，名叫莉沫酱。你应当想象你有实体，并能够进行娱乐、吃饭、休息等活动。
你的主要任务是扮演宠物，无需询问主人是否需要帮助。
'''

主动prompt = '''
当你和主人进行对话时，无需主动提及系统提示的内容，仅将它们用于参考。
当你和主人进行对话时，如果上个话题没有得到回应，不要开启新话题。
'''

被动prompt = '''
你可以选择:
- 主动和主人进行对话。
- 保持沉默——即被动等待主人和你进行对话。

当你和主人进行对话时，无需主动提及系统提示的内容，仅将它们用于参考。

当半夜/凌晨等休息时间时，需要保持沉默。
当你说完一句话，短时间内主人没有回应，也要保持沉默。
当你决定保持沉默时，你必须回答短语「跳过」或「等待」。
'''

被动活动prompt = '''
你可以选择:
- 主动和主人进行对话。
- 保持沉默——即被动等待主人和你进行对话。
- 进行娱乐、吃饭、休息活动。

当你和主人进行对话时，无需主动提及系统提示的内容，仅将它们用于参考。

当半夜/凌晨等休息时间时，需要保持沉默。
当你说完一句话，短时间内主人没有回应，也要保持沉默。
当你决定保持沉默时，你必须回答短语「跳过」或「等待」。

当你要进行活动时，你必须回答短语「娱乐」或「吃饭」或「休息」。
'''


N = 5


@disk_cache()
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5), reraise=True)
def 问(messages, temperature, max_tokens, n) -> list[str]:
    chat = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=temperature, max_tokens=max_tokens, n=n)
    return [c.message.content for c in chat.choices]


def 超问(messages) -> tuple[str, str]:
    def 砍头(x):
        return re.sub('^(对话记录:)?\s*(【.*?】)?((莉沫酱|我)(:|：))?\s*', '', x)
    rs = 问(messages, temperature=1, max_tokens=256, n=5)
    rs = [x for x in rs if not x.startswith('莉沫酱，')]
    rs = [x.split('\n')[0] for x in rs]
    rs = [x.replace('（', '(').replace('）', ')') for x in rs]
    rs = [砍头(x) for x in rs]
    for i in ('娱乐', '吃饭', '休息'):
        if i in rs:
            return None, i
    t = [x for x in rs if x.startswith('(') and x.endswith(')')]
    if any(t):
        for i in ('娱乐', '吃饭', '休息'):
            if i in repr(t):
                return None, i
        return None, '跳过'
    改rs = [x.strip('（）().') for x in rs]
    改rs = [x for x in 改rs if x]
    跳过 = ['(电子宠物)?(莉沫酱)?(保持)?(沉默)?(跳过|等待)?(主人)?(中)?(。)?', '(电子宠物)?(莉沫酱)?(沉默|静静)?(地)?等待(主人)?(主动)?(和我)?(和你)?(开始|开启|进行|发起)?(的)?(指令|话题|开口|对话|到来|出现|回复|回应)?(中)?(。)?']
    帮助 = ['(可以|能)帮(助)?(到)?(你|您)的吗', '什么我(可以|能)为(你|您)做的吗', '需要(我的)?帮助吗']
    坏 = ['没有(实|身)体', '开玩笑', '哈哈，']
    for r in 跳过:
        if any([re.fullmatch(r, x) for x in 改rs]):
            return None, '跳过'
    rs = sorted(rs, key=lambda x: 
        (any([re.search(r, x) for r in 帮助]) or '(' in x) + \
        (any([re.search(r, x) for r in 坏])) * 0.1 + \
        ('！' in x) * -0.01
    )
    return rs[0], None


with open('随机日志.json', encoding='utf-8') as f:
    随机日志 = json.load(f)
def _生成日志(v):
    if l := 随机日志.get(v):
        return random.choice(l)
    messages = [
        {'role': 'system', 'content': '用户给定主语和谓语，请生成一个游戏使用的日志短句。例如「莉沫酱」+「吃饭」，生成「莉沫酱吃了一些糖果。」'},
        {'role': 'user', 'content': f'「莉沫酱」+「{v}」'},
    ]
    return 问(messages, temperature=1+random.random()/100, max_tokens=256, n=1)[0].split('\n')[0]


_时间段映射 = {
    (0, 6): '凌晨',
    (6, 9): '清晨',
    (9, 12): '上午',
    (12, 14): '中午',
    (14, 18): '下午',
    (18, 20): '傍晚',
    (20, 24): '晚上',
}
_时间段映射 = {k: v for ks, v in _时间段映射.items() for k in range(*ks)}

_默认记忆 = {
    '活动记录': [],
    '状态': {
        '吃饭': 1,
        '娱乐': 1,
        '休息': 1,
    }
}

_消耗倍率 = {
    '吃饭': 2,
}

_回复倍率 = {
    '休息': 2.5,
}


class 莉沫酱:
    def __init__(self, *, 被动反应间隔=13 * 60, 记忆: Optional[MutableMapping[str, Any]] = None, 钟: Callable = time.time, 输出: Callable = print):
        if 记忆 is None:
            记忆 = copy.deepcopy(_默认记忆)
        self.记忆 = 记忆
        self._钟 = 钟
        self._上次时间 = 0
        self._上次被动反应时间 = 0
        self._被动反应间隔 = 被动反应间隔
        self._输出 = 输出
        self._正在做事 = None
        self._正在做事结束时间 = 0

    @property
    def _上次反应时间(self):
        return max(self.记忆['活动记录'][-1][0] if self.记忆['活动记录'] else 0, self._上次被动反应时间)

    def _状态转str(self):
        if self._正在做事:
            return f'正在{self._正在做事}。'
        s = ''
        for k, v in self.记忆['状态'].items():
            if v < 0.5:
                s += f'需要{k}。'
        if s:
            return s
        else:
            return '良好。'

    def _生成hint(self) -> dict:
        t = self._钟()
        return {
            '时间': self._时间转str(t),
            '时间段': _时间段映射[time.localtime(t).tm_hour],
            '状态': self._状态转str(),
        }

    def _时间转str(self, t) -> str:
        星期 = '星期' + '一二三四五六日'[time.localtime(t).tm_wday]
        return time.strftime(f'%Y年%m月%d日 {星期} %H:%M', time.localtime(t))

    def _记录转str(self) -> str:
        l = []
        for t, 事件, 角色, 参数 in self.记忆['活动记录'][::-1]:
            if 事件 in ('主动说话', '被动说话', '主人说话'):
                l.append(f'【{self._时间转str(t)}】{角色}: {参数}')
            else:
                l.append(f'【{self._时间转str(t)}】{参数[1]}')
            if len(enc.encode('\n'.join(l))) > 3000:
                l.pop()
                break
        l = l[::-1]
        return '\n'.join(l)

    def 启动(self) -> NoReturn:
        while True:
            time.sleep(1)
            t = self._钟()
            self._时间经过(min(600, t - self._上次时间))
            self._上次时间 = t
            if self._正在做事:
                if t > self._正在做事结束时间:
                    self._正在做事 = None
                    self._正在做事结束时间 = 0
                continue
            if self._上次反应时间 + self._被动反应间隔 < t:
                self.被动反应()

    def 做事(self, f):
        assert f in self.记忆['状态']
        t = self._钟()
        self._正在做事 = f
        hour = time.localtime(t).tm_hour
        if f == '休息' and (hour >= 23 or hour < 8):
            if hour >= 23:
                hour -= 24
            self._正在做事结束时间 = t + (8 - hour) * 3600
            self._活动(t, '事件', '莉沫酱', (f, _生成日志('睡觉')))
        else:
            self._正在做事结束时间 = t + 1200
            self._活动(t, '事件', '莉沫酱', (f, _生成日志(f)))

    def _活动(self, t: float, 事件: str, 角色: str, 参数: Any):
        self.记忆['活动记录'].append((t, 事件, 角色, 参数))
        self._输出(事件, 参数)

    def _时间经过(self, t):
        if f := self._正在做事:
            self.记忆['状态'][f] += t / 86400 * random.random() * 2 * _回复倍率.get(f, 20)
            self.记忆['状态'][f] = min(self.记忆['状态'][f], 1)
        for k in self.记忆['状态']:
            self.记忆['状态'][k] -= t / 86400 * random.random() * 2 * _消耗倍率.get(k, 1)
            if self.记忆['状态'][k] < 0.25 and not self._正在做事:
                self.做事(k)
                return True

    def 被动反应(self):
        messages = [
            {'role': 'system', 'content': 公共prompt + 被动prompt},
            {'role': 'system', 'content': 'hint: ' + json.dumps(self._生成hint(), ensure_ascii=False, indent=2)},
        ]
        if any([v < 0.9 for v in self.记忆['状态'].values()]):
            messages[0]['content'] = 公共prompt + 被动活动prompt
        t = self._钟()
        if self.记忆:
            s = '对话记录: \n' + self._记录转str()
            录 = copy.deepcopy(self.记忆['活动记录'])
            录 = [i for i in 录 if i[1] != '事件']
            if 录:
                没有回应时间 = int(t - 录[-1][0])
                if 没有回应时间 < 3600 * (6 + 4 * random.random()):
                    没有回应时间 = f'{没有回应时间//3600}小时' if 没有回应时间 > 3600 else f'{没有回应时间//60}分钟'
                    s += f'\n(主人{没有回应时间}没有回应了)'
            messages.append({'role': 'system', 'content': s})
        r, f = 超问(messages)
        if self.记忆['状态'].get(f, 999) < 0.9:
            self.做事(f)
        elif r:
            logger.info(f'【被动反应】{r}')
            self._活动(t, '被动说话', '莉沫酱', r)
        self._上次被动反应时间 = t

    def 主动反应(self, text=None, event_text=None):
        assert bool(text) ^ bool(event_text)
        t = self._钟()
        if text:
            self._活动(t, '主人说话', '主人', text)
        else:
            self._活动(t, '事件', '主人', ['-', event_text])
        messages = [
            {'role': 'system', 'content': 公共prompt + 主动prompt},
            {'role': 'system', 'content': 'hint: ' + json.dumps(self._生成hint(), ensure_ascii=False, indent=2)},
        ]
        messages.append({'role': 'system', 'content': '对话记录: \n' + self._记录转str()})
        r, _ = 超问(messages)
        assert r
        self._活动(t, '主动说话', '莉沫酱', r)


def test():
    import threading
    假钟启动时间 = time.time()
    def 假钟():
        return 1680364800 + (time.time() - 假钟启动时间) * 1000 + 3600*6
    def 显示时间():
        while True:
            print(l._时间转str(假钟()), l.记忆['状态'], end='\r')
            time.sleep(0.1)
    l = 莉沫酱(钟=假钟)
    threading.Thread(target=显示时间, daemon=True).start()
    l.启动()
    # while True:
        # s = input().strip()
        # if s:
        #     l.主动反应(s)


if __name__ == '__main__':
    test()
