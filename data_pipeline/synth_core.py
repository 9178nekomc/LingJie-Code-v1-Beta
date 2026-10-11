# -*- coding: utf-8 -*-
"""
LingJie-Code-v1-Pro  —  核心模板注册表（100 个骨架）
数据合成与推理沙箱共用同一套模板真值：
  - 数据生成: meta -> code / expected_stdout
  - 推理管线: TemplateHead 选出 tid 后做结构校验 + 沙箱断言（对拍 expected_stdout）
"""
import random

VARS = ["x", "n", "i", "a", "b", "s", "lst", "num", "count", "name",
        "score", "age", "price", "total", "value", "item", "numbers",
        "data", "text", "word"]
STRS = ["Hello World", "Python", "Hi", "OK", "Good", "abc", "hello",
        "world", "Apple", "Banana", "Test", "Code", "run", "done", "yes"]
NUMS = list(range(1, 51)) + [100, 66, 88, 42, 7, 3, 9, 12, 20]


def rs(rng):  return rng.choice(STRS)
def rn(rng):  return rng.choice(NUMS)
def rv(rng):  return rng.choice(VARS)
def rlist(rng):
    n = rng.randint(3, 5)
    return [rng.randint(1, 20) for _ in range(n)]


def fmt(v):
    if isinstance(v, str):
        return '"%s"' % v
    return str(v)


class T:
    def __init__(self, tid, action, meta_fn, code_fn, out_fn=None, inputs_fn=None,
                 setup=None):
        self.tid = tid
        self.action = action
        self.meta_fn = meta_fn
        self.code_fn = code_fn
        self.out_fn = out_fn          # meta -> stdout str | None(不确定)
        self.inputs_fn = inputs_fn    # meta -> stdin lines list | None
        self.setup = setup            # 沙箱执行前预置代码（如创建测试文件）


TEMPLATES = {}


def reg(tid, action, meta_fn, code_fn, out_fn=None, inputs_fn=None, setup=None):
    TEMPLATES[tid] = T(tid, action, meta_fn, code_fn, out_fn, inputs_fn, setup)


# ========================= print 族 (0-9) =========================
reg(0, "print",
    lambda r: {"action": "print", "data": [rs(r)]},
    lambda m: 'print("%s")' % m["data"][0],
    lambda m: m["data"][0])
reg(1, "print",
    lambda r: {"action": "print", "data": [rn(r)]},
    lambda m: "print(%d)" % m["data"][0],
    lambda m: str(m["data"][0]))
reg(2, "print",
    lambda r: {"action": "print", "data": [rv(r), rn(r)]},
    lambda m: "%s = %d\nprint(%s)" % (m["data"][0], m["data"][1], m["data"][0]),
    lambda m: str(m["data"][1]))
reg(3, "print",
    lambda r: {"action": "print", "data": [rs(r), rv(r), rn(r)]},
    lambda m: ('%s = %d\nprint("%s", %s)'
               % (m["data"][1], m["data"][2], m["data"][0], m["data"][1])),
    lambda m: "%s %d" % (m["data"][0], m["data"][2]))
reg(4, "print",
    lambda r: {"action": "print", "data": [rn(r), rn(r)],
               "body": ["print(var_1 + var_2)"]},
    lambda m: ("var_1 = %d\nvar_2 = %d\nprint(var_1 + var_2)"
               % (m["data"][0], m["data"][1])),
    lambda m: str(m["data"][0] + m["data"][1]))
reg(5, "print",
    lambda r: {"action": "print", "data": [rn(r), rn(r)],
               "body": ["print(var_1 - var_2)"]},
    lambda m: ("var_1 = %d\nvar_2 = %d\nprint(var_1 - var_2)"
               % (m["data"][0], m["data"][1])),
    lambda m: str(m["data"][0] - m["data"][1]))
reg(6, "print",
    lambda r: {"action": "print", "data": [rn(r), rn(r)],
               "body": ["print(var_1 * var_2)"]},
    lambda m: ("var_1 = %d\nvar_2 = %d\nprint(var_1 * var_2)"
               % (m["data"][0], m["data"][1])),
    lambda m: str(m["data"][0] * m["data"][1]))
reg(7, "input",
    lambda r: _c_input_ifelse(r),
    lambda m: ('x = int(input())\nif x < %s:\n    print("%s")\nelse:\n    print("%s")'
               % (m["condition"].split("<")[1].strip(),
                  m["body"][0].split('"')[1], m["body"][1].split('"')[1])),
    lambda m: (m["body"][0].split('"')[1]
               if m["data"][1] < int(m["condition"].split("<")[1])
               else m["body"][1].split('"')[1]),
    lambda m: [str(m["data"][1])])


def _c_input_ifelse(r):
    """复合：输入 + if/else 阈值判断"""
    t, v = rn(r), rn(r)
    a, b = r.choice([("no", "yes"), ("big", "small"), ("low", "high")])
    return {"action": "input", "data": ["x", v], "condition": "x < %d" % t,
            "body": ['if x < %d: print("%s")' % (t, a),
                     'else: print("%s")' % b]}
reg(8, "print",
    lambda r: {"action": "print", "data": [rlist(r)]},
    lambda m: "print(%s)" % m["data"][0],
    lambda m: str(m["data"][0]))
reg(9, "print",
    lambda r: {"action": "print", "data": [rs(r), rs(r)]},
    lambda m: 'print("%s" + " " + "%s")' % (m["data"][0], m["data"][1]),
    lambda m: m["data"][0] + " " + m["data"][1])

# ========================= assign 族 (10-19) =========================
reg(10, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rn(r)]},
    lambda m: "%s = %d" % (m["data"][0], m["data"][1]), None)
reg(11, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rs(r)]},
    lambda m: '%s = "%s"' % (m["data"][0], m["data"][1]), None)
reg(12, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rlist(r)]},
    lambda m: "%s = %s" % (m["data"][0], m["data"][1]), None)
def _c_assign_for(r):
    """复合：赋值 + for 循环打印"""
    v = rn(r)
    return {"action": "assign", "data": ["n", v], "range": "0, %d" % v,
            "body": ["for i in range(%d): print(i)" % v]}

reg(13, "assign",
    lambda r: _c_assign_for(r),
    lambda m: "n = %d\nfor i in range(n):\n    print(i)" % m["data"][1],
    lambda m: "\n".join(str(x) for x in range(m["data"][1])))
reg(14, "assign",
    lambda r: {"action": "assign", "data": ["var_1", "var_2", "var_3",
                                            rn(r), rn(r)],
               "body": ["var_3 = var_1 + var_2"]},
    lambda m: ("var_1 = %d\nvar_2 = %d\nvar_3 = var_1 + var_2"
               % (m["data"][3], m["data"][4])), None)
reg(15, "assign",
    lambda r: {"action": "assign", "data": ["var_1", rs(r)]},
    lambda m: "var_1 = input()", None, lambda m: [m["data"][1]])
reg(16, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rn(r)]},
    lambda m: "%s = int(input())" % m["data"][0], None,
    lambda m: [str(m["data"][1])])
reg(17, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rlist(r)]},
    lambda m: "%s = %s\nn = len(%s)" % (m["data"][0], m["data"][1], m["data"][0]),
    None)
reg(18, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rlist(r)]},
    lambda m: "%s = %s\nm = max(%s)" % (m["data"][0], m["data"][1], m["data"][0]),
    None)
reg(19, "assign",
    lambda r: {"action": "assign", "data": [rv(r), rlist(r)]},
    lambda m: "%s = %s\nm = min(%s)" % (m["data"][0], m["data"][1], m["data"][0]),
    None)

# ========================= if 族 (20-31) =========================
def _if_stdinless(v, x, op):
    return {"action": "if", "condition": "%s %s %d" % (v, op, x),
            "data": [v, x], "body": ['print("big")']}

_OPS = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
        "==": lambda a, b: a == b, ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b, "!=": lambda a, b: a != b}

def _if_code(m):
    v, x = m["data"][0], m["data"][1]
    cond = m["condition"]
    return "%s = %d\nif %s:\n    print(\"big\")" % (v, x, cond)

def _if_out(m):
    v, x = m["data"][0], m["data"][1]
    op = m["condition"].split()[1]
    return "big" if _OPS[op](x, int(m["condition"].split()[2])) else None

for k, op in [(20, ">"), (21, "<"), (22, "=="), (23, ">="), (24, "<="), (25, "!=")]:
    reg(k, "if",
        (lambda op: lambda r: _if_stdinless(rv(r), rn(r), op))(op),
        _if_code, _if_out)

reg(26, "if",
    lambda r: (lambda v, x: {"action": "if", "condition": "%s > %d" % (v, x),
                             "data": [v, x],
                             "body": ['print("big")', 'print("small")']})(rv(r), rn(r)),
    lambda m: ("%s = %d\nif %s:\n    print(\"big\")\nelse:\n    print(\"small\")"
               % (m["data"][0], m["data"][1], m["condition"])),
    lambda m: "big" if m["data"][1] > m["data"][1] else "small")
reg(27, "if",
    lambda r: (lambda v: {"action": "if",
                          "condition": "%s > 1 and %s < 10" % (v, v),
                          "data": [v, rn(r)], "body": ['print("ok")']})(rv(r)),
    lambda m: ("%s = %d\nif %s:\n    print(\"ok\")"
               % (m["data"][0], m["data"][1], m["condition"])),
    lambda m: "ok" if 1 < m["data"][1] < 10 else None)
reg(28, "if",
    lambda r: (lambda v: {"action": "if",
                          "condition": "%s == 1 or %s == 2" % (v, v),
                          "data": [v, rn(r)], "body": ['print("hit")']})(rv(r)),
    lambda m: ("%s = %d\nif %s:\n    print(\"hit\")"
               % (m["data"][0], m["data"][1], m["condition"])),
    lambda m: "hit" if m["data"][1] in (1, 2) else None)
def _c_assign_two_print(r):
    """复合：两次赋值 + 两行打印"""
    a, b = rn(r), rn(r)
    return {"action": "assign", "data": ["x", "y", a, b],
            "body": ["print(x)", "print(y)"]}

reg(29, "assign",
    lambda r: _c_assign_two_print(r),
    lambda m: "x = %d\ny = %d\nprint(x)\nprint(y)"
              % (m["data"][2], m["data"][3]),
    lambda m: "%d\n%d" % (m["data"][2], m["data"][3]))
reg(30, "if",
    lambda r: (lambda v: {"action": "if", "condition": "not %s" % v,
                          "data": [v, 0], "body": ['print("empty")']})(rv(r)),
    lambda m: ("%s = %d\nif %s:\n    print(\"empty\")"
               % (m["data"][0], m["data"][1], m["condition"])),
    lambda m: "empty" if not m["data"][1] else None)
reg(31, "if",
    lambda r: (lambda v, x: {"action": "if", "condition": "%s >= %d" % (v, x),
                             "data": [v, x],
                             "body": ['print("pass")', 'print("fail")']})(rv(r), rn(r)),
    lambda m: ("%s = %d\nif %s:\n    print(\"pass\")\nelse:\n    print(\"fail\")"
               % (m["data"][0], m["data"][1], m["condition"])),
    lambda m: "pass" if m["data"][1] >= int(m["condition"].split()[-1]) else "fail")

# ========================= for 族 (32-45) =========================
reg(32, "for",
    lambda r: {"action": "for", "range": "1, %d" % rn(r), "data": ["i"],
               "body": ["print(i)"]},
    lambda m: "for i in range(%s):\n    print(i)" % m["range"],
    lambda m: "\n".join(str(x) for x in range(
        int(m["range"].split(",")[0]), int(m["range"].split(",")[1]))))
reg(33, "for",
    lambda r: {"action": "for", "range": "1, %d" % rn(r), "data": ["s"],
               "body": ["s += i"]},
    lambda m: "s = 0\nfor i in range(%s):\n    s += i\nprint(s)" % m["range"],
    lambda m: str(sum(range(int(m["range"].split(",")[0]),
                            int(m["range"].split(",")[1])))))
reg(34, "for",
    lambda r: {"action": "for", "range": "1, 4", "data": ["p"],
               "body": ["p *= i"]},
    lambda m: "p = 1\nfor i in range(%s):\n    p *= i\nprint(p)" % m["range"],
    lambda m: str({"1, 4": 6, "1, 5": 24, "2, 5": 60}[m["range"]]))
reg(35, "for",
    lambda r: {"action": "for", "range": "", "data": [rlist(r)],
               "body": ["print(item)"]},
    lambda m: "for item in %s:\n    print(item)" % m["data"][0],
    lambda m: "\n".join(str(x) for x in m["data"][0]))
reg(36, "for",
    lambda r: {"action": "for", "range": "", "data": [rlist(r)],
               "body": ["s += item"]},
    lambda m: "s = 0\nfor item in %s:\n    s += item\nprint(s)" % m["data"][0],
    lambda m: str(sum(m["data"][0])))
reg(37, "for",
    lambda r: {"action": "for", "range": "", "data": [rlist(r)],
               "body": ["m = max(m, item)"]},
    lambda m: "m = 0\nfor item in %s:\n    m = max(m, item)\nprint(m)" % m["data"][0],
    lambda m: str(max(m["data"][0])))
reg(38, "for",
    lambda r: {"action": "for", "range": "0, 10, 2", "data": ["i"],
               "body": ["print(i)"]},
    lambda m: "for i in range(%s):\n    print(i)" % m["range"],
    lambda m: "\n".join(str(x) for x in range(0, 10, 2)))
reg(39, "for",
    lambda r: {"action": "for", "range": "1, %d" % rn(r), "data": ["lst"],
               "body": ["lst.append(i)"]},
    lambda m: "lst = []\nfor i in range(%s):\n    lst.append(i)\nprint(lst)" % m["range"],
    lambda m: str(list(range(int(m["range"].split(",")[0]),
                             int(m["range"].split(",")[1])))))
reg(40, "for",
    lambda r: {"action": "for", "range": "1, 6", "data": ["count"],
               "body": ["count += 1"]},
    lambda m: "count = 0\nfor i in range(%s):\n    count += 1\nprint(count)" % m["range"],
    lambda m: "5")
reg(41, "for",
    lambda r: {"action": "for", "range": "", "data": [rs(r)],
               "body": ["print(ch)"]},
    lambda m: 'for ch in "%s":\n    print(ch)' % m["data"][0],
    lambda m: "\n".join(list(m["data"][0])))
reg(42, "for",
    lambda r: {"action": "for", "range": "%d, %d" % (2, 2 + r.randint(2, 4)),
               "data": [rs(r)], "body": ["print(text)"]},
    lambda m: 'for i in range(%s):\n    print("%s")' % (m["range"], m["data"][0]),
    lambda m: ("\n".join([m["data"][0]] *
                         (int(m["range"].split(",")[1]) - 2))))
reg(43, "for",
    lambda r: {"action": "for", "range": "1, 11", "data": ["i"],
               "body": ["if i % 2 == 0: print(i)"]},
    lambda m: "for i in range(%s):\n    if i %% 2 == 0:\n        print(i)" % m["range"],
    lambda m: "\n".join(str(x) for x in range(2, 11, 2)))
reg(44, "for",
    lambda r: {"action": "for", "range": "", "data": [rlist(r)],
               "body": ["print(lst[i])"]},
    lambda m: "for i in range(len(%s)):\n    print(%s[i])" % (m["data"][0], m["data"][0]),
    lambda m: "\n".join(str(x) for x in m["data"][0]))
reg(45, "for",
    lambda r: {"action": "for", "range": "1, 3", "data": [],
               "body": ["for j in range(1, 3): print(i, j)"]},
    lambda m: ("for i in range(%s):\n"
               "    for j in range(1, 3):\n"
               "        print(i, j)") % m["range"],
    lambda m: "\n".join("%d %d" % (i, j) for i in range(1, 3)
                        for j in range(1, 3)))

# ========================= while 族 (46-53) =========================
reg(46, "while",
    lambda r: {"action": "while", "condition": "i < %d" % rn(r), "data": ["i"],
               "body": ["print(i)", "i += 1"]},
    lambda m: ("i = 0\nwhile %s:\n    print(i)\n    i += 1" % m["condition"]),
    lambda m: "\n".join(str(x) for x in range(int(m["condition"].split("<")[1]))))
reg(47, "while",
    lambda r: {"action": "while", "condition": "i <= %d" % rn(r), "data": ["s"],
               "body": ["s += i", "i += 1"]},
    lambda m: "i = 1\ns = 0\nwhile %s:\n    s += i\n    i += 1\nprint(s)" % m["condition"],
    lambda m: str(sum(range(1, int(m["condition"].split("<=")[1]) + 1))))
reg(48, "while",
    lambda r: {"action": "while", "condition": "n > 0", "data": ["n"],
               "body": ["print(n)", "n -= 1"]},
    lambda m: "n = 3\nwhile %s:\n    print(n)\n    n -= 1" % m["condition"],
    lambda m: "3\n2\n1")
reg(49, "while",
    lambda r: {"action": "while", "condition": "count < 3", "data": ["count"],
               "body": ["count += 1"]},
    lambda m: "count = 0\nwhile %s:\n    count += 1\nprint(count)" % m["condition"],
    lambda m: "3")
reg(50, "while",
    lambda r: {"action": "while", "condition": "True", "data": ["i"],
               "body": ["if i == 3: break", "i += 1"]},
    lambda m: "i = 0\nwhile True:\n    if i == 3:\n        break\n    i += 1\nprint(i)",
    lambda m: "3")
reg(51, "while",
    lambda r: {"action": "while", "condition": "lst", "data": [rlist(r)],
               "body": ["lst.pop()"]},
    lambda m: "lst = %s\nwhile lst:\n    lst.pop()\nprint(lst)" % m["data"][0],
    lambda m: "[]")
reg(52, "while",
    lambda r: {"action": "while", "condition": "n > 0", "data": ["n"],
               "body": ["n -= 2"]},
    lambda m: "n = 6\nwhile %s:\n    n -= 2\nprint(n)" % m["condition"],
    lambda m: "0")
reg(53, "while",
    lambda r: {"action": "while", "condition": "i < 4", "data": ["s"],
               "body": ["s += 1", "i += 1"]},
    lambda m: "i = 0\ns = 0\nwhile %s:\n    s += 1\n    i += 1\nprint(s)" % m["condition"],
    lambda m: "4")

# ========================= input 族 (54-58) =========================
reg(54, "input",
    lambda r: {"action": "input", "data": [rs(r)]},
    lambda m: 'name = input()\nprint(name)',
    lambda m: m["data"][0], lambda m: [m["data"][0]])
reg(55, "input",
    lambda r: {"action": "input", "data": [rv(r), rn(r)]},
    lambda m: '%s = int(input())\nprint(%s + 1)' % (m["data"][0], m["data"][0]),
    lambda m: str(m["data"][1] + 1), lambda m: [str(m["data"][1])])
reg(56, "input",
    lambda r: {"action": "input", "data": [rn(r), rn(r)]},
    lambda m: 'a = int(input())\nb = int(input())\nprint(a + b)',
    lambda m: str(m["data"][0] + m["data"][1]),
    lambda m: [str(m["data"][0]), str(m["data"][1])])
reg(57, "input",
    lambda r: {"action": "input", "data": [rs(r)]},
    lambda m: 'name = input()\nprint("Hello", name)',
    lambda m: "Hello " + m["data"][0], lambda m: [m["data"][0]])
reg(58, "input",
    lambda r: {"action": "input", "data": [rn(r)]},
    lambda m: 'x = int(input())\nif x > 0:\n    print("pos")\nelse:\n    print("neg")',
    lambda m: "pos" if m["data"][0] > 0 else "neg",
    lambda m: [str(m["data"][0])])

# ========================= len/append/pop/sort/max/min/sum (59-75) =========================
reg(59, "len",
    lambda r: {"action": "len", "data": [rlist(r)]},
    lambda m: "print(len(%s))" % m["data"][0],
    lambda m: str(len(m["data"][0])))
reg(60, "len",
    lambda r: {"action": "len", "data": [rs(r)]},
    lambda m: 'print(len("%s"))' % m["data"][0],
    lambda m: str(len(m["data"][0])))
reg(61, "append",
    lambda r: {"action": "append", "data": [rlist(r), rn(r)]},
    lambda m: "lst = %s\nlst.append(%d)\nprint(lst)" % (m["data"][0], m["data"][1]),
    lambda m: str(m["data"][0] + [m["data"][1]]))
reg(62, "append",
    lambda r: {"action": "append", "data": [rlist(r), rs(r)]},
    lambda m: 'lst = %s\nlst.append("%s")\nprint(lst)' % (m["data"][0], m["data"][1]),
    lambda m: str(m["data"][0] + [m["data"][1]]))
reg(63, "append",
    lambda r: {"action": "append", "range": "1, %d" % rn(r), "data": ["lst"],
               "body": ["lst.append(i)"]},
    lambda m: "lst = []\nfor i in range(%s):\n    lst.append(i)\nprint(lst)" % m["range"],
    lambda m: str(list(range(int(m["range"].split(",")[0]),
                             int(m["range"].split(",")[1])))))
reg(64, "pop",
    lambda r: {"action": "pop", "data": [rlist(r)]},
    lambda m: "lst = %s\nlst.pop()\nprint(lst)" % m["data"][0],
    lambda m: str(m["data"][0][:-1]))
reg(65, "pop",
    lambda r: {"action": "pop", "data": [rlist(r)]},
    lambda m: "lst = %s\nlst.pop(0)\nprint(lst)" % m["data"][0],
    lambda m: str(m["data"][0][1:]))
reg(66, "sort",
    lambda r: {"action": "sort", "data": [rlist(r)]},
    lambda m: "lst = %s\nlst.sort()\nprint(lst)" % m["data"][0],
    lambda m: str(sorted(m["data"][0])))
reg(67, "sort",
    lambda r: {"action": "sort", "data": [rlist(r)]},
    lambda m: "new = sorted(%s)\nprint(new)" % m["data"][0],
    lambda m: str(sorted(m["data"][0])))
reg(68, "sort",
    lambda r: {"action": "sort", "data": [rlist(r)]},
    lambda m: "lst = %s\nlst.sort(reverse=True)\nprint(lst)" % m["data"][0],
    lambda m: str(sorted(m["data"][0], reverse=True)))
reg(69, "max",
    lambda r: {"action": "max", "data": [rlist(r)]},
    lambda m: "print(max(%s))" % m["data"][0],
    lambda m: str(max(m["data"][0])))
reg(70, "min",
    lambda r: {"action": "min", "data": [rlist(r)]},
    lambda m: "print(min(%s))" % m["data"][0],
    lambda m: str(min(m["data"][0])))
reg(71, "sum",
    lambda r: {"action": "sum", "data": [rlist(r)]},
    lambda m: "print(sum(%s))" % m["data"][0],
    lambda m: str(sum(m["data"][0])))
reg(72, "max",
    lambda r: {"action": "max", "data": [rn(r), rn(r)]},
    lambda m: "print(max(%d, %d))" % (m["data"][0], m["data"][1]),
    lambda m: str(max(m["data"][0], m["data"][1])))
reg(73, "sum",
    lambda r: {"action": "sum", "range": "1, %d" % rn(r)},
    lambda m: "print(sum(range(%s)))" % m["range"],
    lambda m: str(sum(range(int(m["range"].split(",")[0]),
                            int(m["range"].split(",")[1])))))
reg(74, "min",
    lambda r: {"action": "min", "data": [rn(r), rn(r)]},
    lambda m: "print(min(%d, %d))" % (m["data"][0], m["data"][1]),
    lambda m: str(min(m["data"][0], m["data"][1])))
reg(75, "sum",
    lambda r: {"action": "sum", "data": [rlist(r)],
               "body": ["print(sum(var_1) / len(var_1))"]},
    lambda m: ("var_1 = %s\nprint(sum(var_1) / len(var_1))" % m["data"][0]),
    lambda m: str(sum(m["data"][0]) / len(m["data"][0])))

# ========================= def/return 族 (76-84) =========================
reg(76, "def",
    lambda r: {"action": "def", "data": [rs(r)]},
    lambda m: 'def greet():\n    print("%s")\n\ngreet()' % m["data"][0],
    lambda m: m["data"][0])
reg(77, "def",
    lambda r: {"action": "def", "data": [rn(r), rn(r)]},
    lambda m: "def add(a, b):\n    return a + b\n\nprint(add(%d, %d))" %
              (m["data"][0], m["data"][1]),
    lambda m: str(m["data"][0] + m["data"][1]))
reg(78, "def",
    lambda r: {"action": "def", "data": [rs(r)]},
    lambda m: 'def show():\n    return "%s"\n\nprint(show())' % m["data"][0],
    lambda m: m["data"][0])
reg(79, "def",
    lambda r: {"action": "def", "data": [rn(r)]},
    lambda m: "def f(x):\n    return x * x\n\nprint(f(%d))" % m["data"][0],
    lambda m: str(m["data"][0] * m["data"][0]))
reg(80, "def",
    lambda r: {"action": "def", "data": [rs(r)]},
    lambda m: 'def hello():\n    print("Hello %s")\n\nhello()' % m["data"][0],
    lambda m: "Hello " + m["data"][0])
reg(81, "return",
    lambda r: {"action": "return", "data": [rn(r)]},
    lambda m: "def f():\n    return %d\n\nprint(f())" % m["data"][0],
    lambda m: str(m["data"][0]))
reg(82, "def",
    lambda r: {"action": "def", "data": [rlist(r)]},
    lambda m: "def my_len(lst):\n    return len(lst)\n\nprint(my_len(%s))" % m["data"][0],
    lambda m: str(len(m["data"][0])))
reg(83, "def",
    lambda r: {"action": "def", "range": "1, %d" % rn(r)},
    lambda m: "def f(n):\n    for i in range(n):\n        print(i)\n\nf(%s)" %
              m["range"].split(",")[1].strip(),
    lambda m: "\n".join(str(x) for x in range(int(m["range"].split(",")[1]))))
reg(84, "def",
    lambda r: {"action": "def", "data": [rn(r)]},
    lambda m: ("def check(x):\n"
               "    if x > 0:\n"
               "        return 1\n"
               "    else:\n"
               "        return 0\n\n"
               "print(check(%d))") % m["data"][0],
    lambda m: "1" if m["data"][0] > 0 else "0")

# ========================= import 族 (85-89) =========================
reg(85, "import",
    lambda r: {"action": "import", "data": ["math", "pi"]},
    lambda m: "import math\nprint(math.pi)",
    lambda m: "3.141592653589793")
reg(86, "import",
    lambda r: {"action": "import", "data": ["math", "sqrt", rn(r)]},
    lambda m: "import math\nprint(math.sqrt(%d))" % m["data"][2],
    lambda m: str(math_sqrt(m["data"][2])))
def math_sqrt(x):
    import math as _m
    return _m.sqrt(x)
reg(87, "import",
    lambda r: {"action": "import", "data": ["math", "ceil", rn(r)]},
    lambda m: "import math\nprint(math.ceil(%d / 2))" % m["data"][2],
    lambda m: str(-(-m["data"][2] // 2)))
def _c_input_for(r):
    """复合：输入 + for 循环"""
    v = rn(r)
    return {"action": "input", "data": ["n", v], "range": "0, %d" % v,
            "body": ["for i in range(n): print(i)"]}

reg(88, "input",
    lambda r: _c_input_for(r),
    lambda m: "n = int(input())\nfor i in range(n):\n    print(i)",
    lambda m: "\n".join(str(x) for x in range(m["data"][1])),
    lambda m: [str(m["data"][1])])


def _c_input_max(r):
    """复合：两次输入 + max"""
    v1, v2 = rn(r), rn(r)
    return {"action": "input", "data": ["a", "b", v1, v2],
            "body": ["print(max(a, b))"]}

reg(89, "input",
    lambda r: _c_input_max(r),
    lambda m: "a = int(input())\nb = int(input())\nprint(max(a, b))",
    lambda m: str(max(m["data"][2], m["data"][3])),
    lambda m: [str(m["data"][2]), str(m["data"][3])])

# ========================= open/read/write 族 (90-96) =========================
FILE_TEXT = "Hello A\nWorld B"
_FILE_SETUP = ('open("a.txt", "w").write("%s\\n%s")'
               % (FILE_TEXT.split("\n")[0], FILE_TEXT.split("\n")[1]))
reg(90, "open",
    lambda r: {"action": "open", "data": ["a.txt"]},
    lambda m: 'f = open("a.txt")\nprint(f.read())\nf.close()',
    lambda m: FILE_TEXT, None, setup=_FILE_SETUP)
reg(91, "read",
    lambda r: {"action": "read", "data": ["a.txt"]},
    lambda m: 'f = open("a.txt")\nlines = f.readlines()\nprint(len(lines))\nf.close()',
    lambda m: "2", None, setup=_FILE_SETUP)
reg(92, "write",
    lambda r: {"action": "write", "data": ["a.txt", rs(r)]},
    lambda m: 'f = open("a.txt", "w")\nf.write("%s")\nf.close()\nprint("saved")'
              % m["data"][1],
    lambda m: "saved")
reg(93, "open",
    lambda r: {"action": "open", "data": ["a.txt"]},
    lambda m: 'with open("a.txt") as f:\n    print(f.read())',
    lambda m: FILE_TEXT, None, setup=_FILE_SETUP)
reg(94, "write",
    lambda r: {"action": "write", "data": ["log.txt", rs(r)]},
    lambda m: ('with open("log.txt", "w") as f:\n'
               '    f.write("%s")\nprint("ok")') % m["data"][1],
    lambda m: "ok")
reg(95, "read",
    lambda r: {"action": "read", "data": ["a.txt"]},
    lambda m: 'f = open("a.txt")\ndata = f.read()\nprint(data)\nf.close()',
    lambda m: FILE_TEXT, None, setup=_FILE_SETUP)
reg(96, "write",
    lambda r: {"action": "write", "data": ["out.txt", rn(r)]},
    lambda m: 'f = open("out.txt", "w")\nf.write(str(%d))\nf.close()\nprint("done")'
              % m["data"][1],
    lambda m: "done")

# ========================= 字符串杂项 (97-99) =========================
reg(97, "print",
    lambda r: {"action": "print", "data": [rs(r)]},
    lambda m: 'print("%s".upper())' % m["data"][0],
    lambda m: m["data"][0].upper())
reg(98, "len",
    lambda r: {"action": "len", "data": [rs(r)]},
    lambda m: 'print(len("%s"))' % m["data"][0],
    lambda m: str(len(m["data"][0])))
reg(99, "print",
    lambda r: {"action": "print", "data": [rs(r), rs(r)]},
    lambda m: 'print("%s" + "%s")' % (m["data"][0], m["data"][1]),
    lambda m: m["data"][0] + m["data"][1])

assert len(TEMPLATES) == 100, len(TEMPLATES)


def make_sample(tid, rng):
    """tid -> dict(meta, code, stdout, inputs, setup, action)"""
    t = TEMPLATES[tid]
    meta = t.meta_fn(rng)
    code = t.code_fn(meta)
    stdout = t.out_fn(meta) if t.out_fn else None
    inputs = t.inputs_fn(meta) if t.inputs_fn else None
    return {"tid": tid, "action": t.action, "meta": meta,
            "code": code, "stdout": stdout, "inputs": inputs, "setup": t.setup}
