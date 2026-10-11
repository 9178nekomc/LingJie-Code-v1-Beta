# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  自然语言 Prompt 生成器
输出 (prompt_text, spans)：
  spans = [(char_start, char_end, role)]，role: 0=动作词 1=数据字面量 2=变量名 3=噪声
用于 Layer2 逐 Token 标注、Layer3 动作细分类、Layer1 意图边界。
脏样本（20%）：prompt 中出现中文/非法变量名，metadata 使用规范化英文名。
"""
import random

SYN = {
    "print": ["打印", "输出", "显示", "展示", "print", "show", "打出来",
              "输出一下", "在屏幕上打印", "显示出来"],
    "if": ["如果", "判断", "当", "if", "检查", "要是"],
    "for": ["循环", "遍历", "for", "迭代", "重复", "依次"],
    "while": ["当", "只要", "while", "循环", "反复"],
    "assign": ["赋值", "存到", "定义变量", "设置", "assign", "保存到", "记为"],
    "input": ["输入", "读取用户输入", "input", "获取用户输入", "接收输入"],
    "len": ["长度", "len", "统计长度", "计算长度", "有多少个", "数一数"],
    "append": ["追加", "添加", "加入", "append", "push", "放进"],
    "pop": ["弹出", "移除", "删除", "pop", "去掉"],
    "sort": ["排序", "sort", "从小到大排", "按顺序排", "整理", "升序排列"],
    "max": ["最大值", "最大的", "max", "最大", "最大数"],
    "min": ["最小值", "最小的", "min", "最小", "最小数"],
    "sum": ["总和", "求和", "累加", "sum", "加起来", "一共多少"],
    "def": ["定义函数", "写一个函数", "def", "封装成函数", "新建函数"],
    "return": ["返回", "return", "给出结果", "送回"],
    "import": ["导入", "import", "引入", "加载"],
    "open": ["打开", "open", "载入", "读取文件"],
    "read": ["读取", "读出", "read", "读"],
    "write": ["写入", "写进", "write", "保存到文件"],
}

OP_WORD = {">": ["大于", "超过", "高于", ">"],
           "<": ["小于", "低于", "<"],
           "==": ["等于", "是", "=="],
           ">=": ["大于等于", "不低于", ">="],
           "<=": ["小于等于", "不高于", "<="],
           "!=": ["不等于", "不是", "!="]}

# 脏变量名 -> 规范名（20% 样本注入）
DIRTY_VARS = {"张三": "zhang_san", "李四": "li_si", "我的变量": "my_var",
              "var-1": "var_1", "价格$": "price", "1st": "first",
              "总分": "total_score", "用户 名": "user_name",
              "score!": "score", "我的分数": "my_score"}

PRE = ["", "帮我", "请", "麻烦", "用Python", "写代码", "写个程序", "用Python写代码",
       "帮我写代码", "用python实现", "请用Python", "写一段代码", "编程实现",
       "用代码", "写几行Python"]
POST = ["", "，谢谢", "。", "，要简单一点", "，用最简单的方法", "，谢谢！",
        "，给我完整代码", "，快速实现", "，初学者水平即可"]


def join_parts(parts):
    return "".join(p[0] for p in parts), \
           [(0, len(p[0]), p[1]) for p in parts]


def merge_spans(parts):
    """把相邻 parts 拼成文本并给出连续 span"""
    text = ""
    spans = []
    for t, role in parts:
        spans.append((len(text), len(text) + len(t), role))
        text += t
    return text, spans


def maybe(rng, s, p=0.5):
    return s if rng.random() < p else ""


def dirty_var(rng, legal):
    """以概率返回 (prompt中展示名, 规范名)"""
    if rng.random() < 0.35:
        k = rng.choice(list(DIRTY_VARS))
        return k, DIRTY_VARS[k]
    return legal, legal


# ---------------------------------------------------------------------------
# 按动作生成 prompt parts
# ---------------------------------------------------------------------------
def p_print(rng, meta, tid):
    verb = rng.choice(SYN["print"])
    d = meta["data"]
    if len(d) == 1 and isinstance(d[0], str):
        dtxt = maybe(rng, '"', 0.25) + d[0] + maybe(rng, '"', 0.25)
        core = [(verb, 0), (" ", 3), (dtxt, 1)]
    elif len(d) == 1:
        core = [(verb, 0), (" ", 3), (str(d[0]), 1)]
    elif len(d) == 2:
        core = [(verb, 0), (" ", 3), (str(d[0]), 1), (" 和 ", 3), (str(d[1]), 1)]
    else:
        core = [(verb, 0), (" ", 3), (str(d[0]), 1), (", ", 3),
                (str(d[1]), 1), (", ", 3), (str(d[2] if len(d) > 2 else 0), 1)]
    pats = [
        core,
        [("帮我", 3)] + core,
        core + [("这个内容", 3)],
        [("写代码", 3)] + core,
        [("请", 3)] + core,
        [("用print", 0), (" ", 3)] + [(str(x), 1) for x in d[:1]],
        core + [("出来", 3)],
        [("我要", 3)] + core,
    ]
    return merge_spans(rng.choice(pats))


def p_assign(rng, meta, tid):
    verb = rng.choice(SYN["assign"])
    pats = [
        [("把", 3), (str(meta["data"][1]), 1), (verb, 0), ("给", 3),
         (meta["data"][0], 2)],
        [(meta["data"][0], 2), (verb, 0), ("为", 3), (str(meta["data"][1]), 1)],
        [(meta["data"][0], 2), (" = ", 3), (str(meta["data"][1]), 1)],
        [("定义变量", 3), (meta["data"][0], 2), ("，值为", 3), (str(meta["data"][1]), 1)],
        [("让", 3), (meta["data"][0], 2), ("等于", 3), (str(meta["data"][1]), 1)],
    ]
    return merge_spans(rng.choice(pats))


def cond_text(rng, cond):
    """条件文本 -> parts。支持复合条件与 not"""
    for kw, cn in [(" and ", " 并且 "), (" or ", " 或者 ")]:
        if kw in cond:
            left = cond_text(rng, cond.split(kw)[0])
            right = cond_text(rng, cond.split(kw)[1])
            return left + [(cn, 3)] + right
    toks = cond.split()
    if toks[0] == "not":
        return [(toks[1], 2), (" 为空", 0)]
    if '" in "' in cond:
        q = cond.split('"')
        return [(q[1], 1), (" 在 ", 3), (q[3], 1), (" 里", 0)]
    v, op, n = toks[0], toks[1], " ".join(toks[2:])
    w = rng.choice(OP_WORD[op])
    return [(v, 2), (" ", 3), (w, 0), (" ", 3), (n, 1)]


def p_if(rng, meta, tid):
    verb = rng.choice(SYN["if"])
    cond_parts = cond_text(rng, meta["condition"])
    body = meta["body"][0] if meta.get("body") else 'print("big")'
    act = body.replace("print(", "").replace(")", "").replace('"', "")
    pats = [
        [(verb, 0), (" ", 3)] + cond_parts + [("，就", 3), ("打印", 0), (" ", 3),
                                              (act, 1)],
        [("帮我写：", 3)] + cond_parts + [("时", 3), (verb, 0), ("输出", 0),
                                          (" ", 3), (act, 1)],
        [("写个判断：", 3)] + cond_parts + [("则", 3), ("显示", 0), (act, 1)],
        [("if", 0), (" ", 3)] + [(meta["condition"], 1)] + [(":", 3), ("print", 0),
                                                            ('("%s")' % act, 1)],
    ]
    return merge_spans(rng.choice(pats))


def range_text(rng, rng_str):
    """'1, 5' -> parts '从1到5'"""
    if not rng_str:
        return []
    parts = rng_str.split(",")
    a = parts[0].strip()
    b = parts[1].strip() if len(parts) > 1 else ""
    out = [("从", 3), (a, 1), ("到", 3), (b, 1)]
    if len(parts) > 2:
        out = [("从", 3), (a, 1), ("到", 3), (b, 1), ("，步长", 3),
               (parts[2].strip(), 1)]
    return out


def p_for(rng, meta, tid):
    verb = rng.choice(SYN["for"])
    if meta.get("range"):
        loop_head = [("for i in range(", 3)] + [(meta["range"], 1)] + [("):", 3)]
        rt = range_text(rng, meta["range"])
        var = meta["data"][0] if meta["data"] else "i"
        body_act = meta["body"][0] if meta.get("body") else "print(i)"
        pats = [
            [("写一个", 3), (verb, 0), ("，", 3)] + rt +
            [("，每次", 3), ("打印", 0), (" ", 3), (var, 2)],
            [(verb, 0)] + rt + [("，", 3), ("输出", 0), ("每个数", 3)],
            [("循环", 0)] + rt + [("，依次", 3), ("print", 0), ("(", 3),
                                  (var, 2), (")", 3)],
            loop_head + [("里面", 3), ("print(i)", 0)],
        ]
        return merge_spans(rng.choice(pats))
    # 遍历列表/字符串
    obj = meta["data"][0]
    obj_txt = str(obj)
    pats = [
        [(verb, 0), ("列表 ", 3), (obj_txt, 1), ("，", 3), ("打印", 0), ("每个元素", 3)],
        [("帮我", 3), (verb, 0), (" ", 3), (obj_txt, 1)],
        [("写循环遍历", 3), (obj_txt, 1), ("并输出", 0)],
        [("for item in", 0), (" ", 3), (obj_txt, 1), (":", 3), ("print(item)", 0)],
    ]
    return merge_spans(rng.choice(pats))


def p_while(rng, meta, tid):
    verb = rng.choice(SYN["while"])
    cond = meta["condition"]
    pats = [
        [("写一个", 3), (verb, 0), (cond, 1), ("的循环，", 3), ("打印", 0), ("i", 2)],
        [(verb, 0), (" ", 3), (cond, 1), ("，", 3), ("每次", 3), ("输出", 0), (" i", 2)],
        [("while", 0), (" ", 3), (cond, 1), (":", 3), ("print(i); i += 1", 0)],
        [("用", 3), ("while", 0), ("循环", 3), (cond, 1), ("打印", 0), ("数字", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_input(rng, meta, tid):
    verb = rng.choice(SYN["input"])
    pats = [
        [("读取用户输入", 3), ("，", 3), ("然后", 3), ("打印", 0), ("出来", 3)],
        [("让用户", 3), (verb, 0), ("一个值，", 3), ("再", 3), ("print", 0)],
        [("获取", 3), ("input", 0), ("输入并显示", 0)],
        [("写程序接收", 3), ("输入", 0), ("，", 3), ("打印", 0), ("它", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_len(rng, meta, tid):
    verb = rng.choice(SYN["len"])
    obj = str(meta["data"][0])
    pats = [
        [("计算", 3), (obj, 1), ("的", 3), (verb, 0), ("并打印", 0)],
        [(verb, 0), (" ", 3), (obj, 1)],
        [("打印", 0), (" ", 3), (obj, 1), (" 的", 3), (verb, 0)],
        [("print", 0), ("(", 3), ("len", 0), ("(", 3), (obj, 1), ("))", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_append(rng, meta, tid):
    verb = rng.choice(SYN["append"])
    pats = [
        [("往列表里", 3), (verb, 0), ("元素", 3), ("，", 3), ("打印", 0), ("列表", 3)],
        [(verb, 0), ("一个元素到列表，", 3), ("print", 0), ("列表", 3)],
        [("列表", 3), ("append", 0), ("新元素后", 3), ("打印", 0)],
        [("帮我往数组", 3), (verb, 0), ("数据并显示", 0)],
    ]
    return merge_spans(rng.choice(pats))


def p_pop(rng, meta, tid):
    verb = rng.choice(SYN["pop"])
    pats = [
        [("从列表中", 3), (verb, 0), ("最后一个元素，", 3), ("打印", 0), ("结果", 3)],
        [(verb, 0), ("列表末尾元素", 3)],
        [("帮我", 3), (verb, 0), ("列表的一个元素", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_sort(rng, meta, tid):
    verb = rng.choice(SYN["sort"])
    lst = str(meta["data"][0])
    pats = [
        [("把列表 ", 3), (lst, 1), (" ", 3), (verb, 0), ("并打印", 0)],
        [(verb, 0), ("这个列表：", 3), (lst, 1)],
        [("列表排序", 3), ("，", 3), ("数据是", 3), (lst, 1)],
        [("帮我", 3), (verb, 0), ("列表 ", 3), (lst, 1)],
    ]
    return merge_spans(rng.choice(pats))


def p_maxmin_sum(rng, meta, tid, action):
    verb = rng.choice(SYN[action])
    if action == "sum" and meta.get("range"):
        obj = "1到%s" % meta["range"].split(",")[1].strip()
        pats = [[("计算", 3), (obj, 1), ("的", 3), (verb, 0), ("并打印", 0)],
                [(verb, 0), ("从", 3), (meta["range"].split(",")[0].strip(), 1),
                 ("到", 3), (meta["range"].split(",")[1].strip(), 1)]]
    elif len(meta["data"]) == 2 and isinstance(meta["data"][0], int):
        pats = [[("打印", 0), (" ", 3), (str(meta["data"][0]), 1), (" 和 ", 3),
                 (str(meta["data"][1]), 1), (" 中", 3), ("较大的", 3)],
                [("输出两个数", 3), ("的", 3), (verb, 0)]]
    else:
        obj = str(meta["data"][0])
        pats = [
            [("计算列表 ", 3), (obj, 1), (" 的", 3), (verb, 0), ("并打印", 0)],
            [(verb, 0), (" ", 3), (obj, 1)],
            [("打印列表", 3), ("的", 3), (verb, 0), ("，列表是", 3), (obj, 1)],
            [("求", 3), (obj, 1), ("的", 3), (verb, 0)],
        ]
    return merge_spans(rng.choice(pats))


def p_def(rng, meta, tid):
    pats = [
        [("定义一个", 3), ("函数", 0), ("，", 3), ("打印", 0), ("问候语", 3)],
        [("写一个", 3), ("函数", 0), ("，", 3), ("实现加法并", 3), ("打印", 0), ("结果", 3)],
        [("帮我", 3), ("def", 0), ("一个函数", 3)],
        [("封装一个", 3), ("函数", 0), ("，", 3), ("调用后", 3), ("打印", 0)],
        [("新建一个", 3), ("函数", 0), ("并调用", 3)],
        [("写个方法", 3), ("，", 3), ("运行后", 3), ("打印", 0), ("结果", 3)],
        [("函数", 0), ("练习：", 3), ("写一个并执行", 3)],
        [("用", 3), ("def", 0), ("实现一个小功能", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_return(rng, meta, tid):
    pats = [
        [("写函数", 3), ("返回", 0), ("一个值并打印", 3)],
        [("定义函数", 3), ("return", 0), ("结果", 3)],
        [("函数里", 3), ("返回", 0), ("数字", 3), ("，", 3), ("打印", 0), ("它", 3)],
        [("让函数", 3), ("返回", 0), ("一个数", 3)],
        [("写个带", 3), ("返回值", 0), ("的函数", 3)],
        [("函数处理完", 3), ("return", 0), ("结果", 3)],
        [("返回", 0), ("计算结果", 3), ("并打印", 0)],
    ]
    return merge_spans(rng.choice(pats))


def p_import(rng, meta, tid):
    pats = [
        [("导入", 0), ("math", 1), ("库并", 3), ("打印", 0), ("圆周率", 3)],
        [("import", 0), (" math", 1), ("，", 3), ("输出", 0), ("开平方", 3)],
        [("用math模块计算", 3)],
        [("引入", 0), ("数学库", 3), ("并打印", 0), ("结果", 3)],
        [("导入数学模块", 3), ("，", 3), ("print", 0), ("计算值", 3)],
        [("加载", 0), ("math", 1), ("库", 3)],
        [("用", 3), ("import", 0), ("引入数学工具", 3)],
        [("python怎么导入", 3), ("math", 1), ("并使用", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_open(rng, meta, tid):
    pats = [
        [("打开", 0), ("文件", 3), ("并", 3), ("读取", 0), ("内容", 3)],
        [("open", 0), ("一个文件", 3)],
        [("帮我打开文件", 3), ("读出", 0), ("内容", 3)],
        [("用python打开", 3), ("a.txt", 1)],
        [("打开文本文件", 3), ("显示", 0), ("内容", 3)],
        [("file操作：", 3), ("打开", 0), ("文件", 3)],
        [("读取文件内容", 3), ("，", 3), ("用open", 0)],
    ]
    return merge_spans(rng.choice(pats))


def p_read(rng, meta, tid):
    pats = [
        [("读取", 0), ("文件内容并打印", 3)],
        [("read", 0), ("文件", 3)],
        [("帮我读文件", 3), ("打印", 0), ("出来", 3)],
        [("读出", 0), ("文本内容", 3)],
        [("把文件内容读出来", 3)],
        [("读文件并显示", 0)],
        [("获取文件内容", 3), ("，", 3), ("read", 0)],
    ]
    return merge_spans(rng.choice(pats))


def p_write(rng, meta, tid):
    pats = [
        [("把内容", 3), ("写入", 0), ("文件", 3)],
        [("write", 0), ("文本到文件", 3)],
        [("帮我保存内容到文件", 3)],
        [("把文字写进文件里", 3)],
        [("保存字符串", 3), ("到文件", 3)],
        [("文件写入练习", 3)],
        [("写文件", 0), ("，", 3), ("内容自定义", 3)],
    ]
    return merge_spans(rng.choice(pats))


PROMPT_FN = {
    "print": p_print, "assign": p_assign, "if": p_if, "for": p_for,
    "while": p_while, "input": p_input, "len": p_len, "append": p_append,
    "pop": p_pop, "sort": p_sort, "def": p_def, "return": p_return,
    "import": p_import, "open": p_open, "read": p_read, "write": p_write,
}


# ---------------------------------------------------------------------------
# 复合模板的专用 Prompt（按 tid 覆盖）
# ---------------------------------------------------------------------------
_CN = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

def cn_num(n):
    """1-99 -> 中文数字（十八、二十、四十二）；超界回退阿拉伯数字"""
    n = int(n)
    if n >= 100:
        return str(n)
    if n < 10:
        return _CN[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + _CN[n % 10]
    tens, ones = divmod(n, 10)
    return _CN[tens] + "十" + (_CN[ones] if ones else "")


def p_c_input_ifelse(rng, meta):
    t = meta["condition"].split("<")[1].strip()
    a = meta["body"][0].split('"')[1]
    b = meta["body"][1].split('"')[1]
    tc = cn_num(t) if rng.random() < 0.25 else t
    tc2 = cn_num(t) if rng.random() < 0.25 else t
    pats = [
        [("获取用户输入", 0), ("，", 3), ("如果小于", 0), (tc, 1), ("就输出", 0),
         (a, 1), ("，如果大于", 3), (tc2, 1), ("就输出", 0), (b, 1)],
        [("输入", 0), ("一个数，", 3), ("判断：", 3), ("小于", 0), (tc, 1),
         ("打印", 0), (a, 1), ("，", 3), ("否则打印", 0), (b, 1)],
        [("读取用户输入，", 3), ("如果小于", 0), (tc, 1), ("输出", 0), (a, 1),
         ("，", 3), ("大于", 0), (tc2, 1), ("输出", 0), (b, 1)],
        [("写程序，", 3), ("输入", 0), ("一个数，小于", 3), (tc, 1), ("输出", 0),
         (a, 1), ("，大于就输出", 3), (b, 1)],
    ]
    return merge_spans(rng.choice(pats))


def p_c_assign_for(rng, meta):
    v = str(meta["data"][1])
    vc = cn_num(v) if rng.random() < 0.2 else v
    pats = [
        [("把", 3), (v, 1), ("赋值", 0), ("给", 3), ("n", 2), ("，", 3),
         ("然后", 3), ("循环", 0), ("打印", 0), ("0到", 3), (vc, 1), ("之间每个数", 3)],
        [("定义", 3), ("n", 2), (" = ", 3), (v, 1), ("，", 3), ("用for循环", 0),
         ("打印", 0), ("0到", 3), (vc, 1)],
        [("n", 2), ("等于", 3), (v, 1), ("，", 3), ("循环输出", 0), ("0到", 3),
         (vc, 1)],
    ]
    return merge_spans(rng.choice(pats))


def p_c_assign_two(rng, meta):
    a, b = str(meta["data"][2]), str(meta["data"][3])
    pats = [
        [("把", 3), (a, 1), ("赋值", 0), ("给", 3), ("x", 2), ("，", 3),
         ("把", 3), (b, 1), ("赋值", 0), ("给", 3), ("y", 2), ("，", 3),
         ("然后分别打印", 0)],
        [("定义两个变量", 3), ("x", 2), ("和", 3), ("y", 2), ("，值是", 3),
         (a, 1), ("和", 3), (b, 1), ("，", 3), ("打印", 0), ("它们", 3)],
    ]
    return merge_spans(rng.choice(pats))


def p_c_input_for(rng, meta):
    v = str(meta["data"][1])
    vc = cn_num(v) if rng.random() < 0.2 else v
    pats = [
        [("读取用户输入", 0), ("，", 3), ("循环", 0), ("打印", 0), ("0到", 3),
         (vc, 1), ("之间所有数", 3)],
        [("输入", 0), ("一个数 ", 3), (v, 1), ("，", 3), ("用for循环", 0),
         ("依次打印", 0), ("0到", 3), (vc, 1)],
        [("获取用户输入的数字，", 3), ("循环输出", 0), ("0到", 3), (vc, 1)],
    ]
    return merge_spans(rng.choice(pats))


def p_c_input_max(rng, meta):
    pats = [
        [("输入", 0), ("两个数，", 3), ("打印", 0), ("较大的", 3)],
        [("读取两个输入", 0), ("，", 3), ("输出", 0), ("其中更大的", 3)],
        [("获取两个数", 3), ("，", 3), ("求最大值", 0), ("并打印", 0)],
    ]
    return merge_spans(rng.choice(pats))


TID_PROMPT_FN = {7: p_c_input_ifelse, 13: p_c_assign_for,
                 29: p_c_assign_two, 88: p_c_input_for, 89: p_c_input_max}


def prompt_for_sample(rng, sample, dirty=False):
    """sample = synth_core.make_sample 输出 -> (prompt, spans, prompt_meta)
    dirty=True 时变量名用脏名，返回 prompt_meta（规范化的 meta）"""
    action = sample["action"]
    meta = dict(sample["meta"])
    fn = TID_PROMPT_FN.get(sample["tid"])
    if fn is not None:
        text, spans = fn(rng, meta)
    elif action == "max":
        text, spans = p_maxmin_sum(rng, meta, sample["tid"], "max")
    elif action == "min":
        text, spans = p_maxmin_sum(rng, meta, sample["tid"], "min")
    elif action == "sum":
        text, spans = p_maxmin_sum(rng, meta, sample["tid"], "sum")
    else:
        text, spans = PROMPT_FN[action](rng, meta, sample["tid"])
    if dirty and rng.random() < 0.8:
        k = rng.choice(list(DIRTY_VARS))
        text, spans = inject_dirty(text, spans, k)
        meta = dirty_meta(meta, DIRTY_VARS[k])
    pre = maybe(rng, rng.choice(PRE), 0.7)
    post = maybe(rng, rng.choice(POST), 0.5)
    if pre or post:
        parts = [(pre, 3), (text, None), (post, 3)]
        text2 = pre + text + post
        spans2 = [(0, len(pre), 3)] if pre else []
        off = len(pre)
        for s, e, r in spans:
            spans2.append((s + off, e + off, r))
        if post:
            spans2.append((off + len(text), off + len(text) + len(post), 3))
        text, spans = text2, spans2
    return text, spans, meta


def inject_dirty(text, spans, dirty_name):
    """把文本中的一个变量位置换成脏名（简单实现：附加为变量角色噪声）"""
    # 选取一个角色为 2 的 span 替换；若没有则在前部插入
    tgt = None
    for s, e, r in spans:
        if r == 2:
            tgt = (s, e)
            break
    if tgt is None:
        return dirty_name + text, [(0, len(dirty_name), 2)] + \
               [(s + len(dirty_name), e + len(dirty_name), r) for s, e, r in spans]
    s, e = tgt
    new_text = text[:s] + dirty_name + text[e:]
    dl = len(dirty_name) - (e - s)
    new_spans = []
    for s2, e2, r2 in spans:
        if s2 >= e:
            new_spans.append((s2 + dl, e2 + dl, r2))
        elif e2 <= s:
            new_spans.append((s2, e2, r2))
        else:
            new_spans.append((s, s + len(dirty_name), 2))
    return new_text, new_spans


def dirty_meta(meta, legal):
    """prompt 里的脏名 -> metadata 里的规范名：把 data 中既有 var 换成规范名"""
    m = {k: (list(v) if isinstance(v, list) else v)
         for k, v in meta.items()}
    if "data" in m and m["data"]:
        m["data"] = [legal if isinstance(x, str) and not x.startswith('"')
                     and (x.isalpha() or "_" in x) and x not in ("math", "True")
                     else x for x in m["data"]]
    return m


# ---------------------------------------------------------------------------
# 暴力穷举泛化变体（print / if / for 各 >= 3 万）
# ---------------------------------------------------------------------------
PRINT_TPL = ["打印{d}", "输出{d}", "显示{d}", "把{d}打印出来", "把{d}输出到屏幕",
             "帮我打印{d}", "帮我输出{d}", "请打印{d}", "请输出{d}", "print {d}",
             "print \"{d}\"", "用print输出{d}", "在屏幕上显示{d}", "屏幕打印{d}",
             "写代码打印{d}", "写代码输出{d}", "我要打印{d}", "给我打印{d}",
             "打印一下{d}", "输出一下{d}", "展示{d}", "展示一下{d}", "打出{d}",
             "show {d}", "需要打印{d}", "想打印{d}", "程序打印{d}", "python打印{d}",
             "帮我用print打印{d}", "麻烦打印{d}", "输出这个：{d}", "打印内容{d}",
             "打印字符串{d}", "把{d}显示在屏幕上", "让程序输出{d}", "用代码打印{d}",
             "现在打印{d}", "马上输出{d}", "直接打印{d}", "只打印{d}"]

IF_TPL = ["如果x大于{a}就打印big", "若x超过{a}输出big", "x大于{a}时打印big",
          "当x大于{a}打印big", "判断x大于{a}则打印big", "if x > {a}: print(\"big\")",
          "x > {a} 打印big", "帮我写x大于{a}打印big", "x大于{a}则显示big",
          "要是x高于{a}就输出big", "x大于{a}吗打印big", "检验x大于{a}输出big",
          "x 大于 {a} 打印 big", "如果x > {a} 打印 big", "当x> {a} 打印big",
          "如果变量x大于{a}打印big", "x大于{a}就输出big", "写x大于{a}时打印big",
          "条件x大于{a}打印big", "假设x大于{a}输出big"]

FOR_TPL = ["循环从1到{b}打印每个数", "遍历1到{b}输出i", "for i in range(1, {b}): print(i)",
           "打印1到{b}的所有数", "从1循环到{b}每次打印i", "输出1到{b}之间所有整数",
           "循环打印1到{b}", "for循环1到{b}打印i", "帮我循环打印1到{b}",
           "遍历打印1到{b}的数", "1到{b}依次打印", "把1到{b}都打印出来",
           "迭代1到{b}并输出", "列出1到{b}每个数字", "repeat print 1 to {b}",
           "依次输出1到{b}", "打印区间1到{b}", "循环输出一到{b}", "连续打印1到{b}",
           "求1到{b}每个数并打印"]

VARIANT_NUMS = [str(x) for x in
                list(range(1, 60)) + [66, 88, 100, 77, 99]]
VARIANT_STRS = ["Hello", "Hello World", "Hi", "Python", "你好", "OK", "Test",
                "abc", "world", "yes", "done", "Apple", "Code", "good", "run",
                "你好世界", "morning", "night", "China", "AI", "data", "love",
                "star", "moon", "sun"]


def variant_prompt(rng):
    """穷举变体：模板 × 槽位 × 前后缀修饰，组合空间 >= 10 万"""
    kind = rng.choice(["print", "if", "for"])
    if kind == "print":
        base = rng.choice(PRINT_TPL).format(d=rng.choice(VARIANT_STRS))
    elif kind == "if":
        base = rng.choice(IF_TPL).format(a=rng.choice(VARIANT_NUMS))
    else:
        base = rng.choice(FOR_TPL).format(b=rng.choice(VARIANT_NUMS))
    pre = maybe(rng, rng.choice(PRE), 0.6)
    post = maybe(rng, rng.choice(POST), 0.4)
    return pre + base + post


# ---------------------------------------------------------------------------
# 负样本：非代码请求 / 灰色超纲（触发 Cannot）
# ---------------------------------------------------------------------------
NEG_BASE = [
    "今天天气怎么样", "给我讲个笑话", "推荐几部电影", "Python是什么语言",
    "怎么快速背单词", "帮我写一封辞职信", "把这段话翻译成英文", "1+1等于几",
    "地球到月球有多远", "晚饭吃什么好", "帮我起个网名", "写一首诗",
    "中国的首都是哪里", "帮我订机票", "股票明天会涨吗", "怎么煮红烧肉",
    "解释一下量子力学", "帮我算算这个月花了多少钱", "给我唱首歌", "聊聊人生",
    "如何提高英语口语", "明天要下雨吗", "帮我写周报", "这道数学题怎么解",
    "帮我做PPT", "写一篇作文", "帮我回复老板消息", "健身计划怎么定",
    "哪个牌子的手机好", "帮我查快递", "周末去哪玩", "怎么给孩子起名",
    "教我做蛋糕", "猫粮哪个好", "帮我写检讨书", "如何装修小户型",
    "帮我算生辰八字", "去北京旅游攻略", "怎么挑西瓜", "帮我推荐健身房",
]
GREY_BASE = [
    "写一个爬虫抓取整个网站", "帮我写个贪吃蛇游戏", "实现快速排序算法",
    "写一个完整的网站后端", "破解一下这个WiFi密码", "训练一个图像识别神经网络",
    "批量处理1000个Excel文件", "写一个操作系统内核", "帮我写数据库中间件",
    "实现区块链共识算法", "写一个编译器", "爬取全网商品价格并做价格预测",
    "写个木马程序", "帮我黑进一个网站", "写一套完整的ERP系统",
    "实现分布式消息队列", "写一个深度学习框架", "帮我写手机APP",
    "写一个自动驾驶程序", "做一个人脸识别门禁系统", "写大型3D游戏引擎",
    "实现一个Redis数据库", "帮我写股票量化交易全套系统", "写一个视频剪辑软件",
    "实现图神经网络", "写一个多线程下载器", "帮我实现扫雷游戏",
    "做一个电商推荐系统", "写一个语音识别系统", "实现Manacher算法",
    "写一个区块链钱包", "帮我写聊天机器人全套", "实现操作系统调度器",
    "写一个爬虫代理池", "帮我做数据大屏", "写一个机器翻译模型",
    "实现A*寻路算法", "帮我写自动化运维平台", "做一个短视频去水印工具",
    "写一个CRM客户管理系统", "帮我实现五子棋AI", "写一个NLP分词器",
    "做一个图书管理系统带数据库", "帮我写抢票脚本", "写一个油猴脚本",
    "实现高并发缓存系统", "写一个rpc框架", "帮我做知识图谱",
    "写一个强化学习智能体", "实现一个json解析器",
]
NEG_PRE = ["", "帮我", "请问", "你好，", "老师，", "在吗，", "打扰一下，",
           "想问一下，", "问个问题，", "你好呀，"]
NEG_POST = ["", "？", "。", "！", "呢", "？谢谢", "，谢谢", "，拜托",
            "，在线等", "，速回"]
NEG_FILL = ["", "谢谢", "我是初学者", "麻烦了", "急着用", "帮个忙",
            "科普一下", "详细说说", "简单说说", "举个例子"]


def negative_prompt(rng):
    base = rng.choice(NEG_BASE)
    pre = rng.choice(NEG_PRE)
    post = rng.choice(NEG_POST)
    fill = rng.choice(NEG_FILL)
    sep = "，" if fill and post else ""
    return pre + base + post + sep + fill


def grey_prompt(rng):
    base = rng.choice(GREY_BASE)
    pre = rng.choice(["", "帮我", "请", "麻烦", "用Python", "写个程序", "用代码",
                      "能不能"])
    post = rng.choice(["", "，谢谢", "。", "，要完整的", "，越详细越好",
                       "，功能全一点", "，带注释"])
    fill = rng.choice(NEG_FILL)
    sep = "，" if fill and post else ""
    return pre + base + post + sep + fill
