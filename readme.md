# 莉沫酱电子宠物！

大家有养宠物吗？

宠物很好，它可以陪人玩，这样人就不会无聊了。但是养宠物太贵了，要给它喂食、驱虫、洗澡，每个月要花几百块，而且很麻烦！

于是我想，那就用GPT做一个电子宠物吧，这样大家养宠物就方便了！


## 效果

启动之后你就可以和莉沫酱电子宠物说话啦。

但是和正常版的GPT3.5不同的是，它会在它觉得合适的时候自己找你说话！比如会在早上的时候叫你起来，但是有时候也会在晚上叫你起来——我也不知道为什么，不过既然是宠物那应该是正常的。

然后它饿了会自己去吃东西，不用喂，也会自己去玩和睡觉，真是太方便了！


## 使用方法

它本身是一个class，可以单独运行不过没有前端。推荐的使用方法是接入其他bot框架，仓库里有一个[tg_bot_demo.py](./tg_bot_demo.py)可以参考。

接口是这样:

```py
class 莉沫酱:
    def __init__(self, *, 被动反应间隔=13 * 60, 记忆: Optional[MutableMapping[str, Any]] = None, 钟: Callable = time.time, 输出: Callable = print): ...
    def 启动(self) -> NoReturn: ...
    def 主动反应(self, text=None, event_text=None) -> None : ...
```

构造函数:
- `被动反应间隔`: 莉沫酱每隔这个时间检查一次是否是合适的时候找你说话。
- `记忆`: 这个不是用手填的，而是把上一次程序退出的时候的`self.记忆`填回来。
- `钟`: 莉沫酱内部的时钟，可以通过传一个假的钟来调试。
- `输出`: 当事件触发(如说话)时的回调函数。

启动: 
- 让莉沫酱进入待机状态。饥饿值之类的也会开始随时间消耗。
- 是`while True`循环所以没有返回值。

主动反应: 
- `text`: 和莉沫酱说一句话。
- `event_text`: 告诉莉沫酱你做了一件事。
- 会调用输出的回调函数来输出。


## 一些已知问题

- prompt里提到了「无需询问主人是否需要帮助」，但是它还是会问。

- 你让它10点叫你起床，它是不会叫的，可能是因为是宠物？


## 结束

就这样，大家88，我要回去摸莉沫酱电子宠物了！
