import io
import sys
import csv
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QListWidget, QListWidgetItem, QSpinBox, QSlider, QTabWidget, QGroupBox,
    QMessageBox, QLineEdit, QDateTimeEdit
)

REQUIRED_FIELDS_CN = {
    "name": "姓名",
    "major": "专业",
    "phone": "手机号",
    "title": "职称",
    "email": "邮箱",
}

POSSIBLE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "name": ["姓名", "名字", "Name", "姓名/Name", "员工姓名"],
    "major": ["专业", "专业方向", "Major", "学科", "领域"],
    "phone": ["手机号", "手机", "电话", "联系电话", "Phone", "Mobile", "手机号码", "联系手机"],
    "title": ["职称", "技术职称", "Title", "岗位", "岗位/职称"],
    "email": ["邮箱", "电子邮箱", "Email", "E-mail"],
}

OPTIONAL_ALIASES: Dict[str, List[str]] = {
    "org": ["供职单位", "单位", "工作单位", "所在单位"],
    "reg_org": ["注册单位", "注册单位名称", "注册机构", "注册单位/机构"],
}


def guess_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {k: None for k in REQUIRED_FIELDS_CN.keys()}
    cols = list(df.columns)
    cols_lc = {c: str(c).lower() for c in cols}
    for field, aliases in POSSIBLE_COLUMN_ALIASES.items():
        for a in aliases:
            for c in cols:
                if str(c).strip() == a:
                    mapping[field] = c
                    break
            if mapping[field] is not None:
                break
            a_lc = a.lower()
            for c, c_lc in cols_lc.items():
                if a_lc in c_lc:
                    mapping[field] = c
                    break
            if mapping[field] is not None:
                break
    return mapping


def find_optional(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {"org": None, "reg_org": None}
    cols = list(df.columns)
    cols_lc = {c: str(c).lower() for c in cols}
    for key, aliases in OPTIONAL_ALIASES.items():
        for a in aliases:
            for c in cols:
                if str(c).strip() == a:
                    found[key] = c
                    break
            if found[key] is not None: break
            a_lc = a.lower()
            for c, c_lc in cols_lc.items():
                if a_lc in c_lc:
                    found[key] = c
                    break
            if found[key] is not None: break
    return found


class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("🎯 专家抽样助手 - 原生桌面版")
        self.resize(1180, 780)
        self.df: Optional[pd.DataFrame] = None
        self.mapped_df: Optional[pd.DataFrame] = None
        self.mapping_widgets: Dict[str, QComboBox] = {}
        self.majors_list = QListWidget()
        self.titles_list = QListWidget()
        self.org_avoid_list = QListWidget()
        self.regorg_avoid_list = QListWidget()
        self.optional_cols: Dict[str, Optional[str]] = {"org": None, "reg_org": None}

        root = QVBoxLayout(self)

        # 顶部：项目信息（仿制模式）
        project_group = QGroupBox("项目信息（用于打印显示）")
        pgl = QHBoxLayout()
        self.project_name = QLineEdit(); self.project_name.setPlaceholderText("项目名称")
        self.owner_name = QLineEdit(); self.owner_name.setPlaceholderText("招标人名称")
        self.bid_no = QLineEdit(); self.bid_no.setPlaceholderText("招标编号")
        self.draw_time = QDateTimeEdit(QDateTime.currentDateTime()); self.draw_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        pgl.addWidget(QLabel("项目名称")); pgl.addWidget(self.project_name)
        pgl.addWidget(QLabel("招标人名称")); pgl.addWidget(self.owner_name)
        pgl.addWidget(QLabel("招标编号")); pgl.addWidget(self.bid_no)
        pgl.addWidget(QLabel("抽取时间")); pgl.addWidget(self.draw_time)
        project_group.setLayout(pgl)
        root.addWidget(project_group)

        # 文件区
        file_box = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        btn_open = QPushButton("打开 Excel/CSV")
        btn_open.clicked.connect(self.on_open_file)
        file_box.addWidget(self.file_label)
        file_box.addStretch(1)
        file_box.addWidget(btn_open)
        root.addLayout(file_box)

        # 映射区
        map_group = QGroupBox("字段映射")
        map_layout = QHBoxLayout()
        for key, cn in REQUIRED_FIELDS_CN.items():
            combo = QComboBox(); combo.setMinimumWidth(180); combo.setObjectName(key)
            self.mapping_widgets[key] = combo
            col_box = QVBoxLayout(); col_box.addWidget(QLabel(cn)); col_box.addWidget(combo)
            w = QWidget(); w.setLayout(col_box); map_layout.addWidget(w)
        map_group.setLayout(map_layout)
        root.addWidget(map_group)

        # 筛选区
        filter_group = QGroupBox("筛选（专业/职称）")
        fl = QHBoxLayout()
        self.majors_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.titles_list.setSelectionMode(QListWidget.ExtendedSelection)
        fl.addWidget(self._wrap_with_label("专业(多选)", self.majors_list))
        fl.addWidget(self._wrap_with_label("职称(多选)", self.titles_list))
        filter_group.setLayout(fl)
        root.addWidget(filter_group)

        # 回避区
        avoid_group = QGroupBox("回避（剔除）")
        al = QHBoxLayout()
        self.org_avoid_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.regorg_avoid_list.setSelectionMode(QListWidget.ExtendedSelection)
        al.addWidget(self._wrap_with_label("回避工作单位(多选)", self.org_avoid_list))
        al.addWidget(self._wrap_with_label("回避注册单位(多选)", self.regorg_avoid_list))
        avoid_group.setLayout(al)
        root.addWidget(avoid_group)

        # 抽样区
        sample_group = QGroupBox("随机抽样")
        sl = QHBoxLayout()
        self.count_spin = QSpinBox(); self.count_spin.setRange(0, 100000); self.count_spin.setValue(10)
        self.percent_slider = QSlider(Qt.Horizontal); self.percent_slider.setRange(1, 100); self.percent_slider.setValue(20)
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["按数量", "按百分比"])
        self.btn_sample = QPushButton("开始抽样"); self.btn_sample.clicked.connect(self.on_sample)
        sl.addWidget(QLabel("方式")); sl.addWidget(self.mode_combo)
        sl.addWidget(QLabel("数量")); sl.addWidget(self.count_spin)
        sl.addWidget(QLabel("百分比")); sl.addWidget(self.percent_slider)
        sl.addStretch(1); sl.addWidget(self.btn_sample)
        sample_group.setLayout(sl)
        root.addWidget(sample_group)

        # 导出/打印
        export_box = QHBoxLayout()
        self.btn_export_csv = QPushButton("导出 CSV"); self.btn_export_csv.clicked.connect(lambda: self.export_file("csv"))
        self.btn_export_xlsx = QPushButton("导出 Excel"); self.btn_export_xlsx.clicked.connect(lambda: self.export_file("xlsx"))
        self.btn_print = QPushButton("打印预览"); self.btn_print.clicked.connect(self.on_print)
        export_box.addStretch(1)
        export_box.addWidget(self.btn_export_csv)
        export_box.addWidget(self.btn_export_xlsx)
        export_box.addWidget(self.btn_print)
        root.addLayout(export_box)

    def _wrap_with_label(self, text: str, w: QWidget) -> QWidget:
        box = QVBoxLayout(); box.addWidget(QLabel(text)); box.addWidget(w)
        ww = QWidget(); ww.setLayout(box)
        return ww

    def on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Excel/CSV (*.xlsx *.xls *.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                df = None
                for enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
                    try:
                        df = pd.read_csv(path, encoding=enc)
                        break
                    except Exception:
                        continue
                if df is None:
                    df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return

        df.columns = [str(c).strip() for c in df.columns]
        self.df = df
        self.file_label.setText(Path(path).name)

        # 映射候选填充
        for combo in self.mapping_widgets.values():
            combo.clear(); combo.addItem("<未选择>")
            for c in df.columns:
                combo.addItem(str(c))
        default = guess_mapping(df)
        for key, col in default.items():
            if col is None: continue
            idx = self.mapping_widgets[key].findText(str(col))
            if idx >= 0: self.mapping_widgets[key].setCurrentIndex(idx)

        # 可选列识别 + 回避候选
        self.optional_cols = find_optional(df)
        self.org_avoid_list.clear(); self.regorg_avoid_list.clear()
        if self.optional_cols.get("org"):
            for v in sorted([x for x in df[self.optional_cols["org"]].dropna().unique() if str(x).strip() != ""]):
                self.org_avoid_list.addItem(QListWidgetItem(str(v)))
        if self.optional_cols.get("reg_org"):
            for v in sorted([x for x in df[self.optional_cols["reg_org"]].dropna().unique() if str(v).strip() != ""]):
                self.regorg_avoid_list.addItem(QListWidgetItem(str(v)))

        # 筛选候选
        major = default.get("major"); title = default.get("title")
        self.majors_list.clear(); self.titles_list.clear()
        if major and major in df.columns:
            for v in sorted([x for x in df[major].dropna().unique() if str(x).strip() != ""]):
                self.majors_list.addItem(QListWidgetItem(str(v)))
        if title and title in df.columns:
            for v in sorted([x for x in df[title].dropna().unique() if str(x).strip() != ""]):
                self.titles_list.addItem(QListWidgetItem(str(v)))

    def _build_mapped_df(self) -> Optional[pd.DataFrame]:
        if self.df is None: return None
        mapping: Dict[str, Optional[str]] = {}
        for key, cn in REQUIRED_FIELDS_CN.items():
            text = self.mapping_widgets[key].currentText()
            mapping[key] = None if text == "<未选择>" else text
            if mapping[key] is None:
                QMessageBox.warning(self, "缺少映射", f"请为 {cn} 选择列")
                return None
        result = pd.DataFrame({REQUIRED_FIELDS_CN[k]: self.df[mapping[k]] for k in REQUIRED_FIELDS_CN})
        # 清洗手机号
        phone_cn = REQUIRED_FIELDS_CN["phone"]
        result[phone_cn] = (
            result[phone_cn].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True).str.slice(0, 20)
        )
        return result

    def on_sample(self) -> None:
        mapped = self._build_mapped_df()
        if mapped is None:
            return
        # 过滤
        major_cn = REQUIRED_FIELDS_CN["major"]
        title_cn = REQUIRED_FIELDS_CN["title"]
        majors = {i.text() for i in self.majors_list.selectedItems()}
        titles = {i.text() for i in self.titles_list.selectedItems()}
        filtered = mapped.copy()
        if majors:
            filtered = filtered[filtered[major_cn].isin(majors)]
        if titles:
            filtered = filtered[filtered[title_cn].isin(titles)]
        # 回避剔除
        org_col = self.optional_cols.get("org"); reg_col = self.optional_cols.get("reg_org")
        avoid_orgs = {i.text() for i in self.org_avoid_list.selectedItems()}
        avoid_regs = {i.text() for i in self.regorg_avoid_list.selectedItems()}
        if org_col and org_col in self.df.columns and avoid_orgs:
            # 需要基于原始 df 定位，再映射
            rows_to_drop = self.df[self.df[org_col].isin(list(avoid_orgs))].index
            filtered = filtered.drop(index=filtered.index.intersection(rows_to_drop), errors="ignore")
        if reg_col and reg_col in self.df.columns and avoid_regs:
            rows_to_drop = self.df[self.df[reg_col].isin(list(avoid_regs))].index
            filtered = filtered.drop(index=filtered.index.intersection(rows_to_drop), errors="ignore")

        # 抽样
        mode = self.mode_combo.currentText()
        total = filtered.shape[0]
        if mode == "按数量":
            n = min(total, max(0, int(self.count_spin.value())))
        else:
            n = int(np.floor(total * (self.percent_slider.value() / 100.0)))
            n = min(total, max(1 if total > 0 else 0, n))
        if n <= 0:
            QMessageBox.information(self, "结果", "没有可抽取的数据或抽取数量为0")
            return
        rng = np.random.default_rng()
        idx = rng.choice(filtered.index.values, size=n, replace=False)
        self.mapped_df = filtered.loc[idx]
        QMessageBox.information(self, "结果", f"抽取完成，共 {n} 条。可点击下方导出或打印。")

    def export_file(self, kind: str) -> None:
        if self.mapped_df is None or self.mapped_df.empty:
            QMessageBox.warning(self, "导出", "没有可导出的数据，请先抽样")
            return
        if kind == "csv":
            path, _ = QFileDialog.getSaveFileName(self, "保存CSV", "抽样名单.csv", "CSV (*.csv)")
            if not path: return
            self.mapped_df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存Excel", "抽样名单.xlsx", "Excel (*.xlsx)")
            if not path: return
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                self.mapped_df.to_excel(writer, index=False)

    def on_print(self) -> None:
        if self.mapped_df is None or self.mapped_df.empty:
            QMessageBox.warning(self, "打印", "没有可打印的数据，请先抽样")
            return
        # 简单生成HTML并用系统默认浏览器预览打印
        proj = self.project_name.text().strip()
        owner = self.owner_name.text().strip()
        bid = self.bid_no.text().strip()
        ts = self.draw_time.text()
        html = [
            "<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #999;padding:6px}</style></head><body>",
            f"<h3>项目名称：{proj} ｜ 招标人：{owner} ｜ 招标编号：{bid} ｜ 抽取时间：{ts}</h3>",
            self.mapped_df.to_html(index=False),
            "<script>window.print()</script>",
            "</body></html>",
        ]
        import tempfile
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        fp.write("".join(html).encode("utf-8"))
        fp.close()
        webbrowser.open(f"file:///{fp.name}")


def main() -> None:
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
