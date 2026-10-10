# -*- coding: utf-8 -*-
"""
多 LLM 语料采集器 —— 为 E1/E2/E3/E5/E8 提供跨 LLM 数据

用法：
  1) 在 config.yaml 填写各 LLM 的 base_url / api_key / model
  2) python gen_data.py --llm deepseek --n 1600
  3) 全部采完后运行 python run_all.py

设计要点（对齐论文 §3.1 与评审意见）：
  - 使用 HC3 的 question 作为 prompt，保证与人类/ChatGPT 语料同源、问题级别可比
  - 每个 question 只采集一次回答，保证问题级别隔离（problem-level isolation）
  - 记录 domain 字段，便于 E3 做非 QA 领域扩展
  - 采集结果落盘为 jsonl，任何中断都可续跑（已存在的 question 跳过）
"""
import os, sys, json, io, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import dataio

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
CONFIG = os.path.join(ROOT, 'config.yaml')

DEFAULT_PROMPT = "请回答下面的问题，用中文作答，语气自然，长度适中：\n{question}"


def load_config():
    if not os.path.exists(CONFIG):
        return {}
    try:
        import yaml
        with io.open(CONFIG, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # 无 pyyaml 时做极简解析（只支持顶层 key: value / block）
        cfg, cur = {}, None
        with io.open(CONFIG, encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                if not line.startswith(' '):
                    cur = line.split(':')[0].strip(); cfg[cur] = {}
                elif cur:
                    k, _, v = line.strip().partition(':')
                    cfg[cur][k.strip()] = v.strip().strip('"\'')
        return cfg


def call_llm(cfg, question, prompt_tpl=DEFAULT_PROMPT, max_retry=3):
    """调用 OpenAI 兼容的聊天接口。

    返回 (text, meta)。meta 记录 token 用量与端到端延迟，用于事后核对采集所用的
    模型规模（生成速度 tok/s 是模型量级的间接证据）。
    """
    import requests
    url = cfg['base_url'].rstrip('/') + '/chat/completions'
    headers = {'Authorization': 'Bearer %s' % cfg['api_key'],
               'Content-Type': 'application/json'}
    payload = {'model': cfg['model'],
               'messages': [{'role': 'user', 'content': prompt_tpl.format(question=question)}],
               'temperature': float(cfg.get('temperature', 0.7)),
               'max_tokens': int(cfg.get('max_tokens', 1024))}
    last = None
    for i in range(max_retry):
        try:
            t0 = time.time()
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                body = r.json()
                text = body['choices'][0]['message']['content']
                usage = body.get('usage') or {}
                meta = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'latency_ms': latency_ms,
                        'prompt_tokens': usage.get('prompt_tokens'),
                        'completion_tokens': usage.get('completion_tokens')}
                return text, meta
            last = 'HTTP %d: %s' % (r.status_code, r.text[:200])
        except Exception as e:
            last = str(e)
        time.sleep(2 + 2 * i)
    raise RuntimeError('LLM 调用失败: %s' % last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llm', required=True, help='config.yaml 中的 LLM 名称，如 deepseek/qwen/ernie/llama/gemini/kimi')
    ap.add_argument('--n', type=int, default=1600, help='采集条数')
    ap.add_argument('--domains', default='baike,law,psychology,open_qa,medicine,finance')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None,
                    help='输出路径；默认 data/collected/<llm>_seed<seed>_n<n>.jsonl，'
                         '与下游实验的读取约定一致')
    a = ap.parse_args()

    cfg_all = load_config()
    cfg = (cfg_all.get('llms') or {}).get(a.llm)
    if not cfg or not cfg.get('api_key') or '${' in str(cfg.get('api_key', '')):
        print('[未配置] config.yaml 中缺少 llms.%s 的有效 api_key' % a.llm)
        print('请先填写：', CONFIG)
        sys.exit(2)

    if a.out:
        out_path = a.out
    else:
        out_path = os.path.join(DATA, 'collected',
                                '%s_seed%d_n%d.jsonl' % (a.llm, a.seed, a.n))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with io.open(out_path, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                done.add(d['question'])
        print('续跑：已有 %d 条' % len(done))

    recs = dataio.load_hc3()
    doms = set(a.domains.split(','))
    pool = [r for r in recs if r['domain'] in doms]
    rng = random.Random(a.seed)
    rng.shuffle(pool)

    fout = io.open(out_path, 'a', encoding='utf-8')
    got = 0
    for r in pool:
        if got >= a.n:
            break
        q = r['question']
        if q in done or not q.strip():
            continue
        try:
            ans, meta = call_llm(cfg, q)
        except Exception as e:
            print('  [失败] %s  %s' % (q[:30], e))
            continue
        rec = {'question': q, 'answer': ans, 'domain': r['domain'], 'llm': a.llm}
        rec.update(meta)
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        got += 1
        if got % 50 == 0:
            print('  已采集 %d / %d' % (got, a.n))
    fout.close()
    print('完成：%s 新增 %d 条（累计 %d）' % (out_path, got, len(done) + got))


if __name__ == '__main__':
    main()
