import sys
import os
import json
import re
import fitz  # PyMuPDF
import difflib
import requests
import base64

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QTextEdit, QLabel, QToolBar, QFileDialog, 
                             QMessageBox, QLineEdit, QPushButton, QComboBox, QCheckBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem)
from PyQt6.QtGui import (QAction, QColor, QFont, QImage, QPixmap, QPen, QTextCursor, 
                         QTextCharFormat, QCursor, QTextFormat, QSyntaxHighlighter)
from PyQt6.QtWidgets import QProgressBar
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QEvent, QThread, pyqtSlot
import bisect


# ==========================================
# 0. 全局工具与配置管理
# ==========================================

# 尝试导入本地 OCR
HAS_LOCAL_OCR = False
try:
    from paddleocr import PaddleOCRVL
    HAS_LOCAL_OCR = True
    print("Local PaddleOCR detected.")
except ImportError:
    print("PaddleOCR not found. Local OCR disabled.")

DEFAULT_CONFIG = {
    "pdf_path": "",
    "image_dir": "",
    "start_page": 1,
    "end_page": 1,
    "page_offset": 0,
    "text_path_left": "",
    "text_path_right": "", # 第二版本文本
    "ocr_json_path": "",       # OCR 数据目录
    "regex_left": r"^\*\*(.*?)\*\*",
    "regex_right": r"^([a-zA-Z]*?)",
    "use_pdf_render": False,
    "ocr_api_url": "", # Remote OCR URL
    "ocr_api_token": "", # Remote OCR Token
}

PAGE_PATTERN = re.compile(r"<(\d+)>")

def read_text_to_pages(file_path: str) -> dict[int, str]:
    pages = {}
    if not os.path.exists(file_path): 
        return pages
    try:
        current_page = None
        current_content = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                match = PAGE_PATTERN.fullmatch(line.strip())
                if match:
                    if current_page is not None:
                        pages[current_page] = "\n".join(current_content)
                    current_page = int(match.group(1))
                    current_content = []
                else:
                    current_content.append(line)
            if current_page is not None:
                pages[current_page] = "\n".join(current_content)
    except Exception as e:
        print(f"Read error {file_path}: {e}")
    return pages

def write_pages_to_file(pages: dict[int, str], file_path: str):
    try:
        sorted_pages = sorted(pages.keys())
        with open(file_path, 'w', encoding='utf8') as f:
            for page in sorted_pages:
                text = pages[page]
                f.write(f'<{page}>\n')
                f.write(f'{text}\n')
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Save error: {e}")

# ==========================================
# 1. 自定义编辑器 (支持 Diff 交互) & Highlighter
# ==========================================

class DiffSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.diff_ranges = [] # List of tuples (start, end)
        self.diff_starts = [] # List of start positions for bisect
        self.regex_pattern = None
        
        # 预定义格式
        self.diff_fmt = QTextCharFormat()
        self.diff_fmt.setForeground(QColor("red"))
        self.diff_fmt.setBackground(QColor("#FFEEEE")) # 浅红背景
        
        self.regex_fmt = QTextCharFormat()
        self.regex_fmt.setBackground(QColor("#E0F0FF")) # 浅蓝
        
    def set_diff_data(self, opcodes, is_left):
        self.diff_ranges = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal': continue
            s, e = (i1, i2) if is_left else (j1, j2)
            if s < e:
                self.diff_ranges.append((s, e))
        
        self.diff_ranges.sort() # Ensure sorted
        self.diff_starts = [r[0] for r in self.diff_ranges]
        self.rehighlight()
        
    def set_regex(self, regex_str):
        if not regex_str:
            self.regex_pattern = None
        else:
            try:
                self.regex_pattern = re.compile(regex_str)
            except:
                self.regex_pattern = None
        self.rehighlight()

    def highlightBlock(self, text):
        if not self.diff_ranges and not self.regex_pattern: return
        
        block = self.currentBlock()
        block_start = block.position()
        block_len = len(text)
        block_end = block_start + block_len
        
        # 1. Diff Highlighting (Optimized with Bisect)
        if self.diff_ranges:
            # Find range of opcodes that might overlap with this block.
            # bisect_right returns insertion point after x to maintain order.
            # Any range starting >= block_end essentially cannot overlap (except edge case empty range? invalid here).
            end_idx = bisect.bisect_right(self.diff_starts, block_end)
            
            # Where to start?
            # We need to find the first range that ends > block_start.
            # Since we only have starts list, we can't do exact bisect on ends.
            # But diffs are usually sequential. The ranges ending before block_start 
            # likely have start < block_start.
            # Let's iterate backwards from end_idx or just check a window. 
            # Or just iterate from 0 to end_idx? If end_idx is huge, bad.
            # BUT: In a diff, ranges are strictly increasing.
            # So range[i].end < range[i+1].start usually (unless strictly overlapping... diff opcodes don't overlap).
            # So we can just bisect_right find the UPPER bound.
            # And for lower bound:
            # The range immediately causing overlap could start way before.
            # But opcodes don't overlap.
            # So we can look at bisect_right(starts, block_start) - 1.
            
            start_search = bisect.bisect_right(self.diff_starts, block_start)
            if start_search > 0:
                start_search -= 1 # check the one that started before block
            
            # Safety cap
            count = 0 
            for i in range(start_search, end_idx):
                if count > 1000: break # Safety break
                s, e = self.diff_ranges[i]
                
                # Intersect
                intersect_start = max(s, block_start)
                intersect_end = min(e, block_end)
                
                if intersect_start < intersect_end:
                    rel_start = intersect_start - block_start
                    rel_len = intersect_end - intersect_start
                    self.setFormat(rel_start, rel_len, self.diff_fmt)
                
                count += 1
        
        # 2. Regex Highlighting
        if self.regex_pattern:
            count = 0
            for match in self.regex_pattern.finditer(text):
                if count > 100: break 
                self.setFormat(match.start(), match.end() - match.start(), self.regex_fmt)
                count += 1


class DiffTextEdit(QTextEdit):
    """
    支持 Ctrl+Hover 高亮和 Ctrl+Click 应用补丁的文本框
    """
    # 信号：点击了某个 Diff 块，请求应用到另一侧 (self_index_range, target_text)
    apply_patch_signal = pyqtSignal(tuple, str)
    # 信号：Alt+Click 将本侧内容推送到另一侧 (target_range, my_content)
    push_patch_signal = pyqtSignal(tuple, str)
    
    def __init__(self, side="left"):
        super().__init__()
        self.side = side # 'left' or 'right'
        self.diff_opcodes = [] # 存储 difflib 的 opcodes
        self.other_text_content = "" # 另一侧的完整文本，用于提取
        self.setFont(QFont("Consolas", 11))
        
        # 启用鼠标追踪以支持 Hover
        self.setMouseTracking(True)
        self._hovering_diff = False

    def highlight_line_at_index(self, idx):
        """高亮指定字符索引所在的行"""
        self.blockSignals(True)
        
        # 清除之前的 ExtraSelections (除了 Diff 高亮?)
        # 实际上 diff 高亮是直接作用于 TextCharFormat 的，而 ExtraSelections 是独立的图层
        # 这里仅用于行高亮
        
        cursor = self.textCursor()
        cursor.setPosition(idx)
        
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor("#FFFFAA")) # 淡黄色高亮
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = cursor
        selection.cursor.clearSelection() #只是定位
        
        self.setExtraSelections([selection])
        
        self.blockSignals(False)

    def set_diff_data(self, opcodes, other_text):
        self.diff_opcodes = opcodes
        self.other_text_content = other_text

    def get_opcode_at_position(self, pos):
        """根据鼠标坐标获取对应的 opcode"""
        cursor = self.cursorForPosition(pos)
        idx = cursor.position()
        
        # 遍历 opcodes 查找当前索引是否在差异区间内
        for tag, i1, i2, j1, j2 in self.diff_opcodes:
            if tag == 'equal': continue
            
            # 判断是在左侧还是右侧
            if self.side == 'left':
                # 左侧关注 i1, i2
                # 对于 insert (左侧为空)，范围是 i1==i2，鼠标很难点中，需要容错？
                # 这里主要处理 replace/delete
                if i1 <= idx <= i2:
                    return (tag, i1, i2, j1, j2)
            else:
                # 右侧关注 j1, j2
                if j1 <= idx <= j2:
                    return (tag, i1, i2, j1, j2)
        return None

    def mouseMoveEvent(self, event):
        # 检查是否按住 Ctrl
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            opcode = self.get_opcode_at_position(event.pos())
            if opcode:
                self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
                self._hovering_diff = True
            else:
                self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
                self._hovering_diff = False
        elif modifiers & Qt.KeyboardModifier.AltModifier:
            opcode = self.get_opcode_at_position(event.pos())
            if opcode:
                self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
                self._hovering_diff = True
            else:
                self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
                self._hovering_diff = False
        else:
            self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
            self._hovering_diff = False
        super().mouseMoveEvent(event)

    def clear_highlight(self):
        """清除高亮（ExtraSelections）"""
        self.setExtraSelections([])

    def mousePressEvent(self, event):
        # 处理 Ctrl + Click
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.KeyboardModifier.ControlModifier) and event.button() == Qt.MouseButton.LeftButton:
            opcode = self.get_opcode_at_position(event.pos())
            if opcode:
                self.handle_patch_click(opcode)
                return # 拦截事件，不移动光标

        # 处理 Alt + Click (Push)
        if (modifiers & Qt.KeyboardModifier.AltModifier) and event.button() == Qt.MouseButton.LeftButton:
            opcode = self.get_opcode_at_position(event.pos())
            if opcode:
                self.handle_push_click(opcode)
                return
                
        super().mousePressEvent(event)

    def handle_patch_click(self, opcode):
        tag, i1, i2, j1, j2 = opcode
        
        # 逻辑：点击某侧的差异块，意为“将这一块的内容变成另一侧的样子”
        # 或者“将这一块的内容推送到另一侧”。
        # 通常 Beyond Compare 的逻辑是：点击箭头将当前侧内容覆盖到另一侧。
        # 这里的实现：点击红色区域 -> 将该区域内容替换为另一侧对应区域的内容 (Accept Change)
        
        target_text = ""
        my_range = (0, 0)
        
        if self.side == 'left':
            my_range = (i1, i2)
            # 获取右侧对应文本 (j1:j2)
            target_text = self.other_text_content[j1:j2]
        else:
            my_range = (j1, j2)
            # 获取左侧对应文本 (i1:i2)
            target_text = self.other_text_content[i1:i2]
            
        # 发射信号，由主窗口执行替换操作
        self.apply_patch_signal.emit(my_range, target_text)

    def handle_push_click(self, opcode):
        tag, i1, i2, j1, j2 = opcode
        
        # Logic: Alt+Click = 将“我”的内容推送到“另一侧”
        # 我是 left: 我的内容在 i1:i2, 目标在 j1:j2
        # 我是 right: 我的内容在 j1:j2, 目标在 i1:i2
        
        my_range = (0, 0)
        target_range = (0, 0)
        text_to_push = ""
        current_text = self.toPlainText()
        
        if self.side == 'left':
            my_range = (i1, i2)
            target_range = (j1, j2)
            text_to_push = current_text[i1:i2]
        else:
            my_range = (j1, j2) # Index in right text
            target_range = (i1, i2) # Index in left text
            text_to_push = current_text[j1:j2]
            
        # 发射信号: (目标区间, 要替换成的内容)
        self.push_patch_signal.emit(target_range, text_to_push)


# ==========================================
# 2. 图像画布 (支持缩放、BBox)
# ==========================================

class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        #self.setRenderHint(QPixmap.TransformationMode.SmoothTransformation)
        self.scale_factor = 1.0
        # 拖拽相关
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def load_content(self, pixmap, ocr_data=None):
        self.scene.clear()
        self.highlight_item = None # Fix: Reset C++ object wrapper
        if pixmap:
            self.scene.addPixmap(pixmap)
            self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            if ocr_data:
                self.draw_bboxes(ocr_data)
        self.scale_factor = 1.0
        self.resetTransform()
        
    def draw_bboxes(self, ocr_data):
        pen = QPen(QColor(255, 0, 0, 200))
        pen.setWidth(2)
        
        for item in ocr_data:
            # 兼容 PaddleOCR 格式
            # item 可能是 dict {'bbox':...} (v3代码) 或 list [points, (text, conf)]
            x, y, w, h = 0, 0, 0, 0
            text = ""
            
            if isinstance(item, dict) and 'bbox' in item:
                bbox = item['bbox'] # [x1, y1, x2, y2]
                x, y = bbox[0], bbox[1]
                w, h = bbox[2]-x, bbox[3]-y
                text = item.get('text', '')
            elif isinstance(item, list) and len(item) == 2:
                # Paddle raw: [[[x1,y1],...], ("text", conf)]
                pts = item[0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = min(xs), min(ys)
                w, h = max(xs)-x, max(ys)-y
                text = item[1][0]
            
            rect = QGraphicsRectItem(x, y, w, h)
            rect.setPen(pen)
            rect.setToolTip(text) # 鼠标悬停显示文字
            self.scene.addItem(rect)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom(1.1)
            else:
                self.zoom(0.9)
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom(self, factor):
        self.scale(factor, factor)
        self.scale_factor *= factor

    def ensure_visible_bbox(self, x, y, w, h):
        """确保指定的矩形区域在视图中可见"""
        # 获取场景坐标对应的 Rect
        # 这里 x,y,w,h 已经是场景坐标（基于 Pixmap）
        self.ensureVisible(x, y, w, h, 50, 50) # margin 50

    def set_highlight_bbox(self, x, y, w, h):
        """设置高亮矩形 (蓝色)"""
        # 移除旧的 highlight
        if hasattr(self, 'highlight_item') and self.highlight_item:
            try:
                self.scene.removeItem(self.highlight_item)
            except RuntimeError:
                pass # Already deleted by C++
            self.highlight_item = None
            
        if w > 0 and h > 0:
            pen = QPen(QColor(0, 0, 255, 200)) # Blue
            pen.setWidth(3)
            self.highlight_item = QGraphicsRectItem(x, y, w, h)
            self.highlight_item.setPen(pen)
            self.highlight_item.setZValue(10) # Top layer
            self.scene.addItem(self.highlight_item)
            self.ensure_visible_bbox(x, y, w, h)


# ==========================================
# 2.2 OCR Worker (Async)
# ==========================================

class OCRWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # success, message
    
    def __init__(self, mode, page_list, config, pdf_path=None):
        super().__init__()
        self.mode = mode # 'single' or 'batch'
        self.page_list = page_list # list of page numbers (int)
        self.config = config
        self.pdf_path = pdf_path
        self._is_running = True

    def run(self):
        doc = None
        # Thread-safe PDF opening
        if self.pdf_path and os.path.exists(self.pdf_path):
            try:
                doc = fitz.open(self.pdf_path)
            except: pass
            
        api_url = self.config.get("ocr_api_url")
        token = self.config.get("ocr_api_token")
        
        save_dir = self.config.get("ocr_json_path", "ocr_results")
        if not os.path.exists(save_dir): 
            try: os.makedirs(save_dir)
            except: pass
            
        total = len(self.page_list)
        success_count = 0
        
        for i, page_num in enumerate(self.page_list):
            if not self._is_running: break
            
            try:
                self.progress.emit(f"Processing page {page_num} ({i+1}/{total})...")
                
                # 1. Get Image Data (Bytes)
                img_bytes = None
                
                if doc:
                    try:
                        idx = page_num + self.config['page_offset'] - 1
                        if 0 < idx <= len(doc):
                            page = doc[idx-1]
                            
                            # Try Raw
                            images = page.get_images()
                            if len(images) == 1:
                                xref = images[0][0]
                                base_image = doc.extract_image(xref)
                                img_bytes = base_image["image"]
                            else:
                                # Fallback High DPI
                                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                                img_bytes = pix.tobytes("png")
                    except: pass
                    
                # Try Image Dir if PDF failed
                if not img_bytes:
                    img_dir = self.config.get('image_dir')
                    if img_dir:
                        names = [f"page_{page_num}", f"{page_num}"]
                        exts = [".jpg", ".png", ".jpeg"]
                        for n in names:
                            for e in exts:
                                p = os.path.join(img_dir, n + e)
                                if os.path.exists(p):
                                    with open(p, "rb") as f:
                                        img_bytes = f.read()
                                    break
                            if img_bytes: break
                            
                if not img_bytes:
                    if self.mode == 'single':
                        raise Exception(f"No image found for page {page_num}")
                    else:
                        continue # Skip in batch
                
                # 2. Prepare Request
                file_data = base64.b64encode(img_bytes).decode("ascii")
                headers = {
                    "Authorization": f"token {token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "file": file_data,
                    "fileType": 1,
                    "useDocOrientationClassify": False,
                    "useDocUnwarping": False,
                    "useChartRecognition": False,
                }

                # 3. Send Request
                response = requests.post(api_url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    if self.mode == 'single':
                         raise Exception(f"Remote Error: {response.text}")
                    else:
                         print(f"Page {page_num} Error: {response.text}")
                         continue

                result = response.json().get("result")
                
                # 4. Save
                json_path = os.path.join(save_dir, f"page_{page_num}.json")
                with open(json_path, "w", encoding='utf8') as json_file:
                    json.dump(result, json_file, ensure_ascii=False, indent=2)
                    
                success_count += 1
                
            except Exception as e:
                if self.mode == 'single':
                    self.finished.emit(False, str(e))
                    return
                print(e)
                
        if doc: doc.close()
        self.finished.emit(True, f"Batch OCR Done. {success_count}/{total} processed.")

    def stop(self):
        self._is_running = False


# ==========================================
# 2.5 路径/头部组件 (Beyond Compare Style)
# ==========================================

class FileHeaderWidget(QWidget):
    """
    显示文件路径、浏览按钮、保存按钮
    """
    def __init__(self, parent_window, side="left"):
        super().__init__()
        self.main_window = parent_window
        self.side = side
        self.setFixedHeight(40)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(f"{side} data source path...")
        
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_file)
        
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_file)
        
        layout.addWidget(QLabel(f"{side.upper()}:"))
        layout.addWidget(self.path_edit)
        layout.addWidget(btn_browse)
        layout.addWidget(btn_save)
        
    def set_path(self, path):
        self.path_edit.setText(path)
        
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if filename:
            self.set_path(filename)
            # Update config and reload
            if self.side == "left":
                self.main_window.config['text_path_left'] = filename
            else:
                self.main_window.config['text_path_right'] = filename
            self.main_window.save_config()
            self.main_window.reload_all_data()

    def save_file(self):
        if self.side == "left":
            self.main_window.save_left_data()
        elif self.side == "right":
            self.main_window.save_right_data()


# ==========================================
# 3. 主窗口
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR 校对工具 v4 (PyQt6 Refactor)")
        self.resize(1600, 900)
        
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
        
        # 数据缓存
        self.pages_left = {}  # {page_num: text}
        self.pages_right_text = {} # {page_num: text} (Data Source 2)
        self.current_ocr_data = [] 
        
        self.doc = None # PDF Document
        
        # 脏标记 (Session Sets)
        self.dirty_pages_left = set()
        self.dirty_pages_right = set()
        self._is_updating_diff = False # Recursion Guard
        
        # 初始化界面
        self.init_ui()
        
        # 加载数据
        self.reload_all_data()
        
    def closeEvent(self, event):
        """退出前检查未保存的更改"""
        if self.check_unsaved_changes():
            event.accept()
        else:
            event.ignore()
        
    def load_config(self):
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r", encoding='utf-8') as f:
                    self.config.update(json.load(f))
            except: pass
            
    def save_config(self):
        with open("config.json", "w", encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)

    def init_ui(self):
        # --- 工具栏 ---
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 页码控制
        self.spin_page = QLineEdit()
        self.spin_page.setFixedWidth(50)
        self.spin_page.returnPressed.connect(self.jump_page)
        
        btn_prev = QPushButton("<"); btn_prev.setFixedWidth(30); btn_prev.clicked.connect(self.prev_page)
        btn_next = QPushButton(">"); btn_next.setFixedWidth(30); btn_next.clicked.connect(self.next_page)
        
        toolbar.addWidget(QLabel("页码: "))
        toolbar.addWidget(btn_prev)
        toolbar.addWidget(self.spin_page)
        toolbar.addWidget(btn_next)
        toolbar.addSeparator()
        
        # 数据源选择
        toolbar.addWidget(QLabel(" 右侧数据源: "))
        self.combo_source = QComboBox()
        self.combo_source.addItems(["Text File B", "OCR Results"])
        self.combo_source.currentIndexChanged.connect(lambda: self.load_current_page())
        toolbar.addWidget(self.combo_source)
        
        toolbar.addSeparator()
        
        # 本地 OCR 按钮
        if HAS_LOCAL_OCR:
            btn_ocr = QPushButton("运行本地OCR")
            btn_ocr.clicked.connect(self.run_local_ocr)
            toolbar.addWidget(btn_ocr)
            
        # 远程 OCR 按钮 (仅当配置存在时)
        if self.config.get("ocr_api_url") and self.config.get("ocr_api_token"):
            btn_remote_ocr = QPushButton("运行远程OCR")
            btn_remote_ocr.clicked.connect(self.run_remote_ocr)
            toolbar.addWidget(btn_remote_ocr)
        
        toolbar.addSeparator()
        
        # 保存按钮 (Removing redundant generic save button as requested)
        # btn_save = QPushButton("保存左侧")
        # btn_save.clicked.connect(self.save_left_data)
        # toolbar.addWidget(btn_save)
        
        # 批量 OCR 按钮
        if self.config.get("ocr_api_url") and self.config.get("ocr_api_token"):
             self.btn_batch = QPushButton("OCR所有缺失页面")
             self.btn_batch.clicked.connect(self.run_batch_ocr)
             toolbar.addWidget(self.btn_batch)
        
        btn_export = QPushButton("导出切图")
        btn_export.clicked.connect(self.export_slices)
        toolbar.addWidget(btn_export)

        # --- 主布局 ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 1. 左侧：图片
        self.image_view = ImageCanvas()
        splitter.addWidget(self.image_view)
        
        # 2. 右侧：文本对比
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0,0,0,0)
        
        # 2.1 正则配置区
        regex_layout = QHBoxLayout()
        
        self.regex_input_left = QLineEdit()
        self.regex_input_left.setPlaceholderText("左侧词头正则")
        self.regex_input_left.setText(self.config.get("regex_left", ""))
        self.regex_input_left.editingFinished.connect(self.on_regex_changed)
        
        self.regex_input_right = QLineEdit()
        self.regex_input_right.setPlaceholderText("右侧词头正则")
        self.regex_input_right.setText(self.config.get("regex_right", ""))
        self.regex_input_right.editingFinished.connect(self.on_regex_changed)
        
        regex_layout.addWidget(QLabel("L正则:"))
        regex_layout.addWidget(self.regex_input_left)
        regex_layout.addWidget(QLabel("R正则:"))
        regex_layout.addWidget(self.regex_input_right)
        
        right_layout.addLayout(regex_layout)
        
        # 2.2 文本编辑器区域 (改为带 Header 的布局)
        text_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Side Container ---
        left_container = QWidget()
        left_box = QVBoxLayout(left_container)
        left_box.setContentsMargins(0,0,0,0)
        left_box.setSpacing(0)
        
        self.header_left = FileHeaderWidget(self, "left")
        self.edit_left = DiffTextEdit("left")
        
        left_box.addWidget(self.header_left)
        left_box.addWidget(self.edit_left)
        
        # --- Right Side Container ---
        right_container_widget = QWidget() # Rename to avoid conflict with right_container (outer)
        right_box = QVBoxLayout(right_container_widget)
        right_box.setContentsMargins(0,0,0,0)
        right_box.setSpacing(0)
        
        self.header_right = FileHeaderWidget(self, "right")
        self.edit_right = DiffTextEdit("right")
        
        right_box.addWidget(self.header_right)
        right_box.addWidget(self.edit_right)
        
        # 初始化 Highlighters
        self.highlighter_left = DiffSyntaxHighlighter(self.edit_left.document())
        self.highlighter_right = DiffSyntaxHighlighter(self.edit_right.document())
        
        # 绑定信号
        self.edit_left.textChanged.connect(self.on_text_changed_left)
        self.edit_right.textChanged.connect(self.on_text_changed_right)
        
        # 绑定 Patch 信号
        self.edit_left.apply_patch_signal.connect(lambda r, t: self.apply_patch(self.edit_left, r, t))
        self.edit_right.apply_patch_signal.connect(lambda r, t: self.apply_patch(self.edit_right, r, t))

        # 绑定 Push Patch 信号 (Alt+Click) : 源自 Left -> 改 Right
        self.edit_left.push_patch_signal.connect(lambda r, t: self.apply_patch(self.edit_right, r, t))
        self.edit_right.push_patch_signal.connect(lambda r, t: self.apply_patch(self.edit_left, r, t))
        
        # 绑定滚动同步 (屏蔽默认)
        # self.edit_left.verticalScrollBar().valueChanged.connect(self.sync_scroll_to_right)
        # self.edit_right.verticalScrollBar().valueChanged.connect(self.sync_scroll_to_left)
        # 使用自定义的滚动监听，因为需要判断是否由用户触发
        self.edit_left.verticalScrollBar().valueChanged.connect(lambda v: self.on_scroll(self.edit_left, self.edit_right))
        self.edit_right.verticalScrollBar().valueChanged.connect(lambda v: self.on_scroll(self.edit_right, self.edit_left))
        
        # 绑定光标移动 (高亮对齐 & 自动滚动)
        self.edit_left.cursorPositionChanged.connect(self.on_cursor_left)
        self.edit_right.cursorPositionChanged.connect(self.on_cursor_right)
        
        # 标记是否正在编程滚动，防止死循环
        self._is_program_scrolling = False
        
        # 进度条 (Added to Status Bar)
        # 进度条 (Added to Status Bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        # Finalize Layout
        text_splitter.addWidget(left_container)
        text_splitter.addWidget(right_container_widget)
        right_layout.addWidget(text_splitter)
        
        splitter.addWidget(right_container)
        splitter.setSizes([600, 1000]) # 初始比例

    # ================= 逻辑处理 =================

    def reload_all_data(self):
        # 1. 加载文本
        self.pages_left = read_text_to_pages(self.config['text_path_left'])
        self.pages_right_text = read_text_to_pages(self.config['text_path_right'])
        
        # 2. 加载 PDF
        if self.config['pdf_path'] and os.path.exists(self.config['pdf_path']):
            try:
                self.doc = fitz.open(self.config['pdf_path'])
            except:
                self.doc = None
        
        # 3. Update Headers
        self.header_left.set_path(self.config.get('text_path_left', ''))
        self.header_right.set_path(self.config.get('text_path_right', ''))

        self.load_current_page()

 

    def load_current_page(self):
        # Session-based dirty tracking: No prompts here.
        page_num = self.config.get('start_page', 1)
        try:
            page_num = int(self.spin_page.text())
        except: pass
        
        self.spin_page.setText(str(page_num))
        
        # 1. Load OCR Data
        ocr_data = self.load_ocr_json(page_num)
        self.current_ocr_data = ocr_data # Store for highlighting
        
        # 2. Load Image (High Res)
        if self.doc:
            # Check OCR status
            ocr_state = " (OCR Done)" if ocr_data else " (No OCR)"
            self.statusBar().showMessage(f"Page {page_num} Loaded{ocr_state}")
            
            img_bytes = self.get_best_page_image_bytes(self.doc, page_num)
            if img_bytes:
                img = QImage.fromData(img_bytes)
                pix = QPixmap.fromImage(img)
                self.image_view.load_content(pix, ocr_data)
            else:
                 self.image_view.load_content(None)
        
        # 3. 构建 OCR 映射 (如果存在)
        self.ocr_text_full = ""
        self.ocr_char_map = [] # [(start, end, bbox), ...]
        
        if ocr_data:
            current_idx = 0
            for item in ocr_data:
                text, bbox = "", []
                if isinstance(item, dict):
                    text = item.get('text', '')
                    bbox = item.get('bbox', [])
                elif isinstance(item, list) and len(item) == 2:
                    text = item[1][0]
                    # Parse Paddle points to rect
                    pts = item[0]
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                
                # Append with newline
                chunk = text + "\n"
                length = len(chunk)
                # Map range [current, current+length) -> bbox
                # bbox format for map: [x1, y1, x2, y2] usually
                # We need x,y,w,h or x1,y1,x2,y2. Let's store [x,y,w,h] for easy use
                if len(bbox) == 4:
                     # Check if it is x1,y1,x2,y2 (Paddle usually points.. but here normalized)
                     # My load_ocr_json returns [x,y,w,h] for dict? Let's check load_ocr_json
                     # Wait, load_ocr_json logic for layout parser returns [x,y,x2,y2]??
                     # Let's standardize bbox in process loop below
                     pass
                     
                self.ocr_text_full += chunk
                self.ocr_char_map.append({
                    'start_index': current_idx,
                    'end_index': current_idx + len(text), # exclude newline for click mapping
                    'bbox': bbox
                })
                current_idx += length
        
        is_ocr_mode = (self.combo_source.currentText() == "OCR Results")
        
        # Draw bboxes (Use red for all detected, or maybe lighter if not in OCR mode)
        self.image_view.load_content(pix, ocr_data if ocr_data else [])
        
        # 3. 设置文本
        # 左侧
        left_text = self.pages_left.get(page_num, "")
        
        # 右侧
        right_text = ""
        if is_ocr_mode:
            # Simple join from full text (which includes newlines)
             right_text = self.ocr_text_full
        else:
            right_text = self.pages_right_text.get(page_num, "")

        # 避免触发 textChanged 导致死循环 (以及标记 modified)
        self.edit_left.blockSignals(True)
        self.edit_right.blockSignals(True)
        
        self.edit_left.setPlainText(left_text)
        self.edit_right.setPlainText(right_text)
        
        self.edit_left.blockSignals(False)
        self.edit_right.blockSignals(False)
        
        # 重置脏标记
        self.modified_left = False
        self.modified_right = False
        
        # 4. 执行对比
        self.run_diff()
        
        # 5. 计算 OCR 对齐 (Diff: Left <-> OCR_Full)
        # 如果当前不是 OCR 模式，我们需要额外的 Diff 数据来做映射
        if not is_ocr_mode and self.ocr_text_full:
            self.run_ocr_mapping_diff(left_text)
        else:
            self.ocr_diff_opcodes = self.edit_left.diff_opcodes # 复用主 Diff (如果右侧就是OCR)

    def get_best_page_image_bytes(self, doc, page_num):
        """Extract High-Res or Raw image from PDF"""
        try:
             idx = page_num + self.config['page_offset'] - 1
             if 0 < idx <= len(doc):
                 page = doc[idx - 1]
                 
                 # 1. Try Extract Raw Image (scanned PDF)
                 try:
                     images = page.get_images()
                     if len(images) == 1:
                         xref = images[0][0]
                         base_image = doc.extract_image(xref)
                         return base_image["image"]
                 except: pass # some implementation might fail
                 
                 # 2. Fallback: High DPI Render
                 pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
                 return pix.tobytes("png")
        except: pass
        return None

    def get_page_pixmap(self, page_num):
        """Helper for image cropping etc (still needed?) -> Refactor to use get_best..."""
        if self.doc:
            b = self.get_best_page_image_bytes(self.doc, page_num)
            if b:
                img = QImage.fromData(b)
                return QPixmap.fromImage(img)
        img_dir = self.config['image_dir']
        if img_dir and os.path.exists(img_dir):
            # 尝试 page_1.jpg 或 1.jpg
            names = [f"page_{page_num}", f"{page_num}"]
            exts = [".jpg", ".png", ".jpeg"]
            for n in names:
                for e in exts:
                    p = os.path.join(img_dir, n + e)
                    if os.path.exists(p):
                        return QPixmap(p)
        return None

    def load_ocr_json(self, page_num):
        """加载 PaddleOCR 格式 JSON"""
        path = self.config['ocr_json_path']
        f_path = os.path.join(path, f"page_{page_num}.json")
        if not os.path.exists(f_path):
            # 尝试直接数字
            f_path = os.path.join(path, f"{page_num}.json")
        
        if os.path.exists(f_path):
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 简单适配逻辑：
                    # 如果是标准 Paddle list: [[points, (text, conf)], ...]
                    # 如果是 layout parser: data['fullContent']... (需要解析)
                    
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and ("fullContent" in data or "layoutParsingResults" in data):
                        # 简化解析 PaddleOCR VL
                        res = []
                        if "fullContent" in data:
                            data = data["fullContent"]
                        if "layoutParsingResults" in data:
                            data = data["layoutParsingResults"][0]
                        blocks = data.get("prunedResult", {}).get("parsing_res_list", [])
                        for b in blocks:
                            if b.get('block_label') == 'text':
                                res.append({
                                    'text': b.get('block_content'),
                                    'bbox': b.get('block_bbox')
                                })
                        return res
            except Exception as e:
                print(f"JSON Load error: {e}")
        return []

    # ================= Diff 核心 =================
    
    def run_ocr_mapping_diff(self, left_text):
        """计算 Left Text 到 OCR Full Text 的 Diff，用于坐标映射"""
        if not self.ocr_text_full: 
            self.ocr_diff_opcodes = []
            return
            
        matcher = difflib.SequenceMatcher(None, left_text, self.ocr_text_full, autojunk=False)
        self.ocr_diff_opcodes = matcher.get_opcodes()

    def run_diff(self):
        if self._is_updating_diff: return
        self._is_updating_diff = True
        
        try:
            text_l = self.edit_left.toPlainText()
            text_r = self.edit_right.toPlainText()
            
            matcher = difflib.SequenceMatcher(None, text_l, text_r, autojunk=False)
            opcodes = matcher.get_opcodes()
            
            # 将 diff 数据传递给编辑器，供交互使用
            self.edit_left.set_diff_data(opcodes, text_r)
            self.edit_right.set_diff_data(opcodes, text_l)
            
            # 渲染颜色 (使用 Highlighter)
            # Block signals to prevent feedback loop
            self.edit_left.blockSignals(True)
            self.edit_right.blockSignals(True)
            self.highlighter_left.set_diff_data(opcodes, is_left=True)
            self.highlighter_right.set_diff_data(opcodes, is_left=False)
            self.edit_left.blockSignals(False)
            self.edit_right.blockSignals(False)
            
            # 渲染词头正则
            self.highlighter_left.set_regex(self.config.get("regex_left"))
            self.highlighter_right.set_regex(self.config.get("regex_right"))
            
            # 如果不在 OCR 模式，更新 OCR Mapping
            if self.combo_source.currentText() != "OCR Results" and self.ocr_text_full:
                self.run_ocr_mapping_diff(text_l)
        finally:
            self._is_updating_diff = False


    def highlight_editor(self, editor, opcodes, is_left):
        # Deprecated: Logic moved to DiffSyntaxHighlighter
        pass

    def highlight_regex(self, editor, regex_str):
        # Deprecated: Logic moved to DiffSyntaxHighlighter
        pass

    # ================= 交互 =================

    def apply_patch(self, editor, rng, target_text):
        """应用 Diff 补丁：将 range 区间的内容替换为 target_text"""
        start, end = rng
        cursor = editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(target_text)
        # 插入后 textChanged 会触发，自动重新 diff

    def check_unsaved_changes(self):
        """
        检查未保存 (Exit Only). 如果有，弹窗提示。
        """
        if self.dirty_pages_left or self.dirty_pages_right:
            msg = "Unsaved changes in:\n"
            if self.dirty_pages_left: msg += "- Left Text\n"
            if self.dirty_pages_right: msg += "- Right Text\n"
            msg += "Do you want to save?"
            
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return False
            elif reply == QMessageBox.StandardButton.Yes:
                if self.dirty_pages_left: self.save_left_data()
                if self.dirty_pages_right: self.save_right_data()
                return True
        return True

    def on_text_changed_left(self):
        if not self._is_updating_diff:
            try:
                p = int(self.spin_page.text())
                self.dirty_pages_left.add(p)
            except: pass
        self.deferred_run_diff()
        
    def on_text_changed_right(self):
        if not self._is_updating_diff:
            try:
                p = int(self.spin_page.text())
                self.dirty_pages_right.add(p)
            except: pass
        self.deferred_run_diff()

    def deferred_run_diff(self):
        # 每次变动都实时同步内存 (old behavior)
        try:
            page_num = int(self.spin_page.text())
            # self.pages_left[page_num] = self.edit_left.toPlainText() # 不要直接改 Source，这违背了“未保存”逻辑
            # 我们应该仅在 Save 时写入 self.pages_left / Write Dict。
            # BUT: 程序的逻辑是 page_left 是 dict 缓存。如果切换页面不保存，这些改动就丢了。
            # 原逻辑是实时写入 pages_left，但 pages_left 只是内存。save_left_data 才是写入磁盘。
            # 这里的 Unsaved Prompt 指的是写入磁盘。
            # 所以：我们继续保持实时更新 memory dict 以便运行 diff，但是 save_left_data 负责持久化。
            self.pages_left[page_num] = self.edit_left.toPlainText()
            if self.combo_source.currentText() == "Text File B":
                 self.pages_right_text[page_num] = self.edit_right.toPlainText()
        except: pass
        self.run_diff()

    def on_regex_changed(self):
        self.config["regex_left"] = self.regex_input_left.text()
        self.config["regex_right"] = self.regex_input_right.text()
        self.save_config()
        self.run_diff()

    # ================= 交互增强 (Sync & Highlight) =================

    def get_mapped_index(self, idx, is_left_source):
        """获取索引映射"""
        opcodes = self.edit_left.diff_opcodes
        mapped_idx = -1
        
        for tag, i1, i2, j1, j2 in opcodes:
            # src range, dst range
            s1, s2 = (i1, i2) if is_left_source else (j1, j2)
            d1, d2 = (j1, j2) if is_left_source else (i1, i2)
            
            if s1 <= idx <= s2:
                if tag == 'equal':
                    offset = idx - s1
                    mapped_idx = d1 + offset
                    if mapped_idx > d2: mapped_idx = d2
                else:
                    mapped_idx = d1
                break
        return mapped_idx

    def on_scroll(self, source, target):
        """基于内容的对齐滚动"""
        if self._is_program_scrolling: return
        
        # 获取 Source 视口最顶端的字符索引
        # cursorForPosition(0,0) 获取的是 visual line 的开始
        # 为了更准确，可以取一点 margin，比如 (5, 5)
        top_cursor = source.cursorForPosition(source.viewport().rect().topLeft())
        src_idx = top_cursor.position()
        
        is_left = (source == self.edit_left)
        
        # 映射到 Target
        dst_idx = self.get_mapped_index(src_idx, is_left)
        
        if dst_idx >= 0:
            self._is_program_scrolling = True
            
            # 计算目标位置的 Y 坐标
            # 方法：找到 dst_idx 所在的 block，获取其 bounding rect
            doc = target.document()
            block = doc.findBlock(dst_idx)
            layout = doc.documentLayout()
            
            # blockBoundingRect 返回的是相对于文档的坐标
            block_rect = layout.blockBoundingRect(block)
            
            # 也可以更精细：如果是 wrap 过的长行，blockBoundingRect 是整个 block 的
            # 我们只需要大致对齐 block 即可
            target_y = block_rect.y()
            
            target.verticalScrollBar().setValue(int(target_y))
            
            self._is_program_scrolling = False

    def request_highlight_other(self, source_editor, idx):
        """根据当前光标位置，高亮另一侧对应位置"""
        # 1. 清除本侧高亮 (避免双方都有黄色条，只保留光标在当前侧，高亮在另一侧)
        source_editor.clear_highlight()
        
        # 2. 确定方向
        is_left_source = (source_editor == self.edit_left)
        target_editor = self.edit_right if is_left_source else self.edit_left
        
        mapped_idx = self.get_mapped_index(idx, is_left_source)
        
        if mapped_idx >= 0:
            target_editor.highlight_line_at_index(mapped_idx)
        else:
            # 如果没找到映射（比如超出范围），也清除对面
            target_editor.clear_highlight()

    def check_auto_scroll_bbox(self, editor, idx):
        """检查是否需要自动滚动图片 (针对 PaddleOCR 结果)"""
        if not self.current_ocr_data: return
        
        # 如果是 Right Editor 且处于 OCR 模式，直接用行号 (旧逻辑保留，简单快速)
        is_ocr_mode = (self.combo_source.currentText() == "OCR Results")
        if editor == self.edit_right and is_ocr_mode:
            self._handle_right_editor_ocr_scroll(editor, idx)
            return

        # 如果是 Left Editor (或者 Right Editor 非 OCR 模式)
        # 使用 Diff Mapping 映射到 OCR Index
        
        target_ocr_idx = -1
        
        # 1. 确定 Mapping Source
        if editor == self.edit_left:
             # Use ocr_diff_opcodes (Left -> OCR)
             opcodes = getattr(self, 'ocr_diff_opcodes', [])
             src_idx = idx
             
             # Map src_idx to ocr_idx
             for tag, i1, i2, j1, j2 in opcodes:
                 if i1 <= src_idx <= i2:
                     if tag == 'equal':
                         target_ocr_idx = j1 + (src_idx - i1)
                         if target_ocr_idx > j2: target_ocr_idx = j2
                     else:
                         target_ocr_idx = j1
                     break
        
        # 2. Find BBox for target_ocr_idx
        if target_ocr_idx >= 0 and hasattr(self, 'ocr_char_map'):
            for mapping in self.ocr_char_map:
                # {start_index, end_index, bbox}
                # Use loose check: if index falls in line range
                if mapping['start_index'] <= target_ocr_idx <= mapping['end_index'] + 1: # +1 includes newline
                    bbox = mapping['bbox']
                    # standardize bbox to x,y,w,h
                    x, y, w, h = 0,0,0,0
                    if len(bbox) == 4:
                         # 假设是 x1,y1,x2,y2 (LayoutParser) 
                         # 或者 x,y,w,h (Paddle Dict)?
                         # 需要根据 load_ocr_json 的实际输出来定。
                         # 查看 load_ocr_json:
                         #   Paddle Dict: 'bbox': [x1, y1, x2, y2]
                         #   Paddle List: calculated [minx, miny, maxx, maxy]
                         # So it is consistently [x1, y1, x2, y2] in my code logic above (lines 535-544 modify it logic? Wait)
                         # line 535 logic: `bbox = [min(xs), min(ys), max(xs), max(ys)]` which is [x1, y1, x2, y2]
                         # line 532 logic: `bbox = item.get('bbox', [])`. standard layout parser is [x,y,w,h]? No usually [x1,y1,x2,y2]. 
                         # Let's assume [x1, y1, x2, y2]
                         
                         x, y = bbox[0], bbox[1]
                         w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                         
                    self.image_view.set_highlight_bbox(x, y, w, h)
                    return

    def _handle_right_editor_ocr_scroll(self, editor, idx):
        # 原有的逻辑：按行号
        cursor = editor.textCursor()
        line_num = cursor.blockNumber() # 0-indexed
        if 0 <= line_num < len(self.current_ocr_data):
            # ... (Logic to find bbox from current_ocr_data list) ...
            # Reuse logic implicitly via creating a map first? 
            # Actually, let's just reuse the generic map logic if possible, 
            # BUT right editor in OCR mode is strictly line-synced.
            item = self.current_ocr_data[line_num]
            x, y, w, h = 0,0,0,0
            
            # ... Copy paste old logic ...
            if isinstance(item, dict) and 'bbox' in item:
                b = item['bbox']
                x, y, w, h = b[0], b[1], b[2]-b[0], b[3]-b[1]
            elif isinstance(item, list) and len(item) == 2:
                pts = item[0]
                # ...
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x, y = min(xs), min(ys)
                w, h = max(xs)-x, max(ys)-y
                
            self.image_view.set_highlight_bbox(x, y, w, h)

    def on_cursor_left(self):
        idx = self.edit_left.textCursor().position()
        self.request_highlight_other(self.edit_left, idx)
        # 增加：检查左侧光标对应的 BBox
        self.check_auto_scroll_bbox(self.edit_left, idx)

    def on_cursor_right(self):
        idx = self.edit_right.textCursor().position()
        self.request_highlight_other(self.edit_right, idx)
        self.check_auto_scroll_bbox(self.edit_right, idx)

    # ================= 功能 =================

    def prev_page(self):
        # No check needed
        try:
            p = int(self.spin_page.text())
            self.spin_page.setText(str(p - 1))
            self.load_current_page()
        except: pass

    def next_page(self):
        # No check needed
        try:
            p = int(self.spin_page.text())
            self.spin_page.setText(str(p + 1))
            self.load_current_page()
        except: pass
        
    def jump_page(self):
        # No check needed
        self.load_current_page()

    def save_left_data(self):
        path = self.config.get('text_path_left')
        if not path:
             # Provide Save As?
             path, _ = QFileDialog.getSaveFileName(self, "Save Left", "", "Text (*.txt)")
             if path: 
                 self.config['text_path_left'] = path
        
        if path:
            write_pages_to_file(self.pages_left, path)
            self.dirty_pages_left.clear()
            QMessageBox.information(self, "保存", f"Left data saved to {path}")

    def save_right_data(self):
        if self.combo_source.currentText() != "Text File B":
            QMessageBox.warning(self, "Error", "Right side is not a text file.")
            return

        path = self.config.get('text_path_right')
        if not path:
             path, _ = QFileDialog.getSaveFileName(self, "Save Right", "", "Text (*.txt)")
             if path: 
                 self.config['text_path_right'] = path
        
        if path:
            write_pages_to_file(self.pages_right_text, path)
            self.dirty_pages_right.clear()
            QMessageBox.information(self, "保存", f"Right data saved to {path}")

    def run_local_ocr(self):
        if not HAS_LOCAL_OCR: return
        page_num = int(self.spin_page.text())
        img_path = ""
        
        # 导出当前页面图片为临时文件供 OCR
        pix = self.get_page_pixmap(page_num)
        if not pix:
            QMessageBox.warning(self, "Error", "没有图片可供 OCR")
            return
            
        temp_img = "temp_ocr.jpg"
        pix.save(temp_img)
        
        try:
            self.statusBar().showMessage("OCR Running...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            ocr = PaddleOCRVL()
            result = ocr.predict(temp_img)
            
            # 转换为标准格式并保存
            # result 结构 [[[x,y]..], (text, conf)]
            data = result[0] if result else []
            
            # 保存到 json
            save_dir = self.config['ocr_json_path']
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            json_path = os.path.join(save_dir, f"page_{page_num}.json")
            
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("OCR Done.")
            
            # 切换模式并刷新
            self.combo_source.setCurrentText("OCR Results")
            self.load_current_page()
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(e)
            QMessageBox.critical(self, "OCR Error", str(e))

            QMessageBox.critical(self, "OCR Error", str(e))

    def run_remote_ocr(self):
        """运行单页远程 OCR (Async)"""
        api_url = self.config.get("ocr_api_url")
        token = self.config.get("ocr_api_token")
        if not api_url or not token:
            QMessageBox.warning(self, "Config", "Missing URL/Token")
            return

        try:
            page_num = int(self.spin_page.text())
        except: return
        
        # Disable button?
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.start_ocr_thread('single', [page_num])

    def run_batch_ocr(self):
        """批量 OCR / Cancel"""
        # Toggle Logic: Cancel
        if hasattr(self, 'ocr_thread') and self.ocr_thread and self.ocr_thread.isRunning():
            self.ocr_thread.stop()
            self.btn_batch.setText("Stopping...")
            self.btn_batch.setEnabled(False)
            return

        # Start Logic
        api_url = self.config.get("ocr_api_url")
        token = self.config.get("ocr_api_token")
        if not api_url or not token:
            QMessageBox.warning(self, "Config", "Missing URL/Token")
            return
            
        start = self.config.get("start_page", 1)
        end = self.config.get("end_page", 100)
        
        missing_pages = []
        save_dir = self.config.get("ocr_json_path", "ocr_results")
        
        for p in range(start, end + 1):
            if not os.path.exists(os.path.join(save_dir, f"page_{p}.json")):
                missing_pages.append(p)
                
        if not missing_pages:
            QMessageBox.information(self, "Info", "No missing OCR pages found.")
            return
            
        # ret = QMessageBox.question(self, "Batch OCR", f"Found {len(missing_pages)} missing pages. Start?", 
        #                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        # if ret == QMessageBox.StandardButton.Yes:
        
        # Direct Start with Cancel Option
        self.btn_batch.setText("Cancel OCR")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(missing_pages))
        self.progress_bar.setValue(0)
        
        self.start_ocr_thread('batch', missing_pages)

    def start_ocr_thread(self, mode, pages):
        self.ocr_thread = OCRWorker(mode, pages, self.config, self.doc) # Pass raw doc? no, pass path or handle in main
        # Passing self.doc to thread is risky if main thread accesses it. 
        # But OCRWorker only reads. However, fitz doc might be not thread safe.
        # Better: OCRWorker opens its own doc (via path).
        # We need to pass pdf_path.
        
        # Wait, get_best_page_image_bytes needs doc. 
        # If we pass doc, we must ensure main thread doesn't use it or lock it.
        # Safest: Let Worker open file.
        
        self.ocr_thread = OCRWorker(mode, pages, self.config, self.config.get('pdf_path'))
        self.ocr_thread.progress.connect(self.on_ocr_progress)
        self.ocr_thread.finished.connect(self.on_ocr_finished)
        self.ocr_thread.start()

    def on_ocr_progress(self, msg):
        self.statusBar().showMessage(msg)
        if hasattr(self, 'ocr_thread') and self.ocr_thread.mode == 'batch':
             val = self.progress_bar.value()
             self.progress_bar.setValue(val + 1)

    def on_ocr_finished(self, success, msg):
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(msg, 5000)
        
        # Reset Batch UI
        if hasattr(self, 'btn_batch'):
            self.btn_batch.setText("OCR所有缺失页面")
            self.btn_batch.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            if self.ocr_thread.mode == 'single':
                # Reload current page
                self.combo_source.setCurrentText("OCR Results")
                self.load_current_page()
            else:
                QMessageBox.information(self, "Batch Done", msg)
        else:
            if "Program interrupted" not in msg: # Don't error on manual stop
                 QMessageBox.critical(self, "OCR Failed", msg)
        
        self.ocr_thread = None

    def export_slices(self):
        """如果当前是 OCR 模式且有 BBox 数据，则切割"""
        if self.combo_source.currentText() != "OCR Results" or not self.current_ocr_data:
            QMessageBox.warning(self, "Warning", "当前不是 OCR 模式或没有 OCR 数据，无法切割。")
            return
            
        out_dir = "output_slices"
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        page_num = self.spin_page.text()
        pix = self.get_page_pixmap(int(page_num))
        if not pix: return
        
        img = pix.toImage()
        
        count = 0
        for i, item in enumerate(self.current_ocr_data):
            # 获取 bbox
            x, y, w, h = 0, 0, 0, 0
            if isinstance(item, dict):
                b = item['bbox']
                x, y, w, h = b[0], b[1], b[2]-b[0], b[3]-b[1]
            elif isinstance(item, list):
                pts = item[0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = min(xs), min(ys)
                w, h = max(xs)-x, max(ys)-y
            
            # 切割
            rect = img.copy(int(x), int(y), int(w), int(h))
            rect.save(os.path.join(out_dir, f"{page_num}_{i}.jpg"))
            count += 1
            
        QMessageBox.information(self, "Export", f"已导出 {count} 张切片到 {out_dir}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 全局字体设置，防止显示过小
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())