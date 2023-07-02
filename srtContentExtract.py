
import sys
import pysrt
from datetime import datetime


def main():
    # 参数1：输入的srt，参数2：输出的txt，默认datetime.txt
    srt_file = sys.argv[1]
    if len(sys.argv) < 3:
        output_file = f"./doc/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    else:
        output_file = sys.argv[2]
    subs = pysrt.open(srt_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sub in subs:
            # print(f"时间：{sub.start} --> {sub.end}")
            print(f"文本：{sub.text}")
            f.write(f"{sub.text},")

if __name__ == "__main__":
    main()

