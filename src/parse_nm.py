# -*- coding: utf-8 -*-
# 只读 data/nm_train.xlsx（第一列为中文标签；无表头），解析并导出到 data/meta_labels/
import os, re, json
import numpy as np
import pandas as pd

# ========= 写死路径 =========
XLSX_PATH = os.path.join("data", "nm_train.xlsx")
OUT_DIR   = os.path.join("data", "meta_labels")
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 固定映射与候选词 =========
_TOP_MAP = {"非漏-音频":0, "渗漏-音频":1, "阀门漏水-音频":2}
_TOP_KEYWORDS = [("非漏",0), ("渗漏",1), ("阀门漏水",2), ("阀门",2)]
_MATERIAL_LIST  = ["球墨铸铁","球墨","铸铁","镀锌钢","钢","不锈钢","PE","PVC","PPR","UPVC","复合","混凝土","铜"]
_COMPONENT_LIST = ["阀门","接头","管体","三通","弯头","法兰","水表","消火栓","堵头","变径","密封","支架"]

# ========= 简单工具 =========
def _normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("（","(").replace("）",")").replace("，",",").replace("：",":").replace("；",";")
    for ch in ["／","/","|","、","·"]:
        s = s.replace(ch, "\\")
    for ch in ["—","–","－","~","～"]:
        s = s.replace(ch, "-")
    return re.sub(r"\s+", "", s)

_num_pat = r"([0-9]+(?:\.[0-9]+)?)"
def _pick_float(pat, text): m = re.search(pat, text); return float(m.group(1)) if m else np.nan
def _pick_int  (pat, text): m = re.search(pat, text); return int(m.group(1))   if m else np.nan
def _pick_from_list(cands, text, default="Unknown"):
    for w in cands:
        if w in text: return w
    return default

def _parse_top(text: str) -> str:
    for k in _TOP_MAP.keys():
        if k in text: return k
    for kw, _ in _TOP_KEYWORDS:
        if kw in text:
            return f"{kw}-音频" if kw != "非漏" else "非漏-音频"
    return "非漏-音频"  # 保守默认

def parse_one(raw_text: str) -> dict:
    t = _normalize_text(raw_text)
    # 顶层 & 二分类
    top = _parse_top(t)
    top_id = _TOP_MAP.get(top, 0)
    is_leak = 0 if top_id == 0 else 1
    # 非漏子类：仅对非漏细分（0 常规；1 周期/脉冲；其它=-1）
    nonleak_sub = -1
    if top_id == 0:
        if any(k in t for k in ["周期性","周期","脉冲","脉冲干扰"]):
            nonleak_sub = 1
        else:
            nonleak_sub = 0
    # 可选字段（有就解析）
    leak_rate = _pick_float(r"漏量"+_num_pat, t)
    dn        = _pick_int  (r"DN(\d+)", t)
    depth     = _pick_float(r"埋深"+_num_pat, t)
    dist      = _pick_float(r"距离"+_num_pat, t)
    material  = _pick_from_list(_MATERIAL_LIST,  t, "Unknown")
    component = _pick_from_list(_COMPONENT_LIST, t, "Unknown")
    return {
        "raw": raw_text,
        "top": top,
        "top_id": int(top_id),
        "is_leak": int(is_leak),
        "nonleak_sub": int(nonleak_sub),   # -1=非非漏；0=常规非漏；1=周期/脉冲非漏
        "leak_rate": leak_rate, "DN": dn, "depth": depth, "distance": dist,
        "material": material, "component": component,
    }

def main():
    # 读取 Excel：首行当数据（header=None），用第一个工作表；第一列为整行中文标签
    sheets = pd.read_excel(XLSX_PATH, sheet_name=None, header=None)
    first_sheet = next(iter(sheets))
    df = sheets[first_sheet]
    tag_series = df.iloc[:, 0].astype(str).fillna("")
    print(f"[INFO] 读取工作表: {first_sheet} | 行数={len(tag_series)}")

    # 解析
    recs = [parse_one(s) for s in tag_series.tolist()]
    out = pd.DataFrame.from_records(recs)

    # 统计
    print("\n[STATS] 顶层分布（top）:")
    print(out["top"].value_counts(dropna=False))
    print("\n[STATS] is_leak 分布:")
    print(out["is_leak"].value_counts(dropna=False))
    print("\n[STATS] 非漏子类（-1=非非漏; 0=常规非漏; 1=周期/脉冲）:")
    print(out["nonleak_sub"].value_counts(dropna=False))

    # 保存
    csv_out = os.path.join(OUT_DIR, "nm_parsed.csv")
    out.to_csv(csv_out, index=False, encoding="utf-8-sig")
    np.save(os.path.join(OUT_DIR, "meta_is_leak.npy"),     out["is_leak"].to_numpy(np.int64))
    np.save(os.path.join(OUT_DIR, "meta_top_id.npy"),      out["top_id"].to_numpy(np.int64))
    np.save(os.path.join(OUT_DIR, "meta_nonleak_sub.npy"), out["nonleak_sub"].to_numpy(np.int64))
    with open(os.path.join(OUT_DIR, "meta_label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"top_map": _TOP_MAP}, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {csv_out}")
    print("[SAVED] meta_is_leak.npy, meta_top_id.npy, meta_nonleak_sub.npy, meta_label_map.json")

if __name__ == "__main__":
    main()
