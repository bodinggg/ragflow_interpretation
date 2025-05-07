import logging

import os
import random
import re
import sys
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer

import numpy as np
print(f'starting...')
import pdfplumber
import trio

import xgboost as xgb

from huggingface_hub import snapshot_download

from PIL import Image
from pypdf import PdfReader as pdf2_read
print(f'Recognizer 待加载')
from recognizer import Recognizer
from ocr import OCR
print(f'模块加载完成')

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class PdfParser:
    def __init__(self):
        self.ocr = OCR()
    
    def _has_color(self, o):
        #print(f'[_has_color] 调用')
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and \
                    o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True
    
    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        print(f'[__ocr] 调用')
        start = timer()
        bxs = self.ocr.detect(np.array(img), device_id)
        logging.info(f"__ocr detecting boxes of a image cost ({timer() - start}s)")

        start = timer()
        if not bxs:
            self.boxes.append([])
            return
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [{"x0": b[0][0] / ZM, "x1": b[1][0] / ZM,
              "top": b[0][1] / ZM, "text": "", "txt": t,
              "bottom": b[-1][1] / ZM,
              "page_number": pagenum} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
            self.mean_height[-1] / 3
        )

        # merge chars in the same rect
        for c in Recognizer.sort_Y_firstly(
                chars, self.mean_height[pagenum - 1] // 4):
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != ' ':
                self.lefted_chars.append(c)
                continue
            if c["text"] == " " and bxs[ii]["text"]:
                if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", bxs[ii]["text"][-1]):
                    bxs[ii]["text"] += " "
            else:
                bxs[ii]["text"] += c["text"]

        logging.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        start = timer()
        boxes_to_reg = []
        img_np = np.array(img)
        for b in bxs:
            if not b["text"]:
                left, right, top, bott = b["x0"] * ZM, b["x1"] * \
                                         ZM, b["top"] * ZM, b["bottom"] * ZM
                b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
                boxes_to_reg.append(b)
            del b["txt"]
        texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
        logging.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[-1] == 0:
            self.mean_height[-1] = np.median([b["bottom"] - b["top"]
                                              for b in bxs])
        self.boxes.append(bxs)
    
    
    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback = None):
        print(f'[__images__] 调用')
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        start = timer()
        with (pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))) as pdf:
            # 实例捕捉pdf
            self.pdf = pdf
            # 实例捕获转换的图片列表（pdf转）
            self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                                enumerate(self.pdf.pages[page_from:page_to])]
            try:
                # 利用pdfplumber中的dedupe_chars方法，通过比较文本内容与定位坐标去重。最终是一个二维list
                self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
            except Exception as e:
                logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                self.page_chars = [[] for _ in range(page_to - page_from)]  # If failed to extract, using empty list instead.
            self.total_page = len(self.pdf.pages)
        
        self.outlines = []
        with (pdf2_read(fnm)) as pdf:
            self.pdf = pdf
            
            outlines = self.pdf.outline
            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth +1)
            dfs(outlines, 0)
            print(f'self.outlines is {self.outlines}')
        self.is_english = [re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(
            random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i]))))) for i in
            range(len(self.page_chars))]
        if sum([1 if e else 0 for e in self.is_english]) > len(
                self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False
        
        def __img_ocr(i, id, img, chars):
            print(f'[__img_ocr] 调用')
            j = 0
            while j + 1 < len(chars):
                if chars[j]["text"] and chars[j + 1]["text"] \
                        and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"]) \
                        and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"],
                                                                       chars[j]["width"]) / 2:
                    chars[j]["text"] += " "
                j += 1

            self.__ocr(i + 1, img, chars, zoomin, id)

            if callback and i % 6 == 5:
                callback(prog=(i + 1) * 0.6 / len(self.page_images), msg="")
        
        def __img_ocr_launcher():
            print(f'[__img_ocr_launcher] 调用')
            def __ocr_preprocess():
                print(f'[__ocr_preprocess] 调用')
                chars = self.page_chars[i] if not self.is_english else []
                self.mean_height.append(
                    np.median(sorted([c["height"] for c in chars])) if chars else 0
                )
                self.mean_width.append(
                    np.median(sorted([c["width"] for c in chars])) if chars else 8
                )
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars

            for i, img in enumerate(self.page_images):
                chars = __ocr_preprocess()
                __img_ocr(i, 0, img, chars)
        
        __img_ocr_launcher()          
        print(f'[__images__] __img_ocr_launcher() 调用结束')
        if not self.is_english and not any(
                [c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                                        "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)
                    
        print(f'[__images__] 调用结束，total page is {self.total_page}')
        
    def __call__(self, fnm, need_image = True, zoomin=3, return_html=False):
        print(f'[__call__] 调用')
        self.__images__(fnm)

if __name__ == "__main__":
    print(f'开始执行')
    psr = PdfParser()
    psr_result = psr(sys.argv[1])
    print(psr_result)