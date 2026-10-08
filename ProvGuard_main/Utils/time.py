# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime, timezone
import pytz

# ==== 1. 配置路径与时区 ====
folder = r"D:\PC_project\Attack_investgate\src\GCL\data\trace"        # 替换为你的目录
pattern_prefix = "ta1-trace-e3"       # 文件名前缀
end_prefix = ".txt"
us_tz = pytz.timezone("America/New_York")  # 需要的美国时区

result = {}

def ns_to_est(ts_ns: str) -> str:
    """纳秒时间戳 -> 美国东部时间字符串"""
    ts_sec = int(ts_ns) / 1e9
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    return dt_utc.astimezone(us_tz).strftime("%Y-%m-%d %H:%M:%S.%f %Z")

for fname in os.listdir(folder):
    if fname.startswith(pattern_prefix) and fname.endswith(end_prefix):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            # 读取所有行首尾
            first_line = None
            last_line  = None

            for line in f:
                if not first_line:
                    first_line = line.strip()
                last_line = line.strip()   # 始终更新到最后一行

        # 如果文件为空则跳过
        if not first_line:
            continue

        # 如果文件只有一行，则 last_line == first_line
        if not last_line:
            last_line = first_line

        # 取时间戳列
        first_ts = first_line.split("\t")[-1]
        last_ts  = last_line.split("\t")[-1]

        result[fname] = {
            "start_utc_ns": first_ts,
            "end_utc_ns":   last_ts,
            "start_us_time": ns_to_est(first_ts),
            "end_us_time":   ns_to_est(last_ts)
        }

# 保存 JSON
out_path = os.path.join(folder, "darpa_tc_time_ranges.json")
with open(out_path, "w", encoding="utf-8") as out:
    json.dump(result, out, indent=4)

print(f"统计完成，结果保存在: {out_path}")
