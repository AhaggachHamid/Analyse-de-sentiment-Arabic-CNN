import os
import glob

files = glob.glob("/vercel/sandbox/*.txt")
files.sort()

for f in files:
    print(f"\n{'#'*80}")
    print(f"FILE: {f}")
    print('#'*80)
    with open(f, "r", encoding="utf-8") as fh:
        content = fh.read()
    print(content)
