import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit.components.v1 import html as st_html


# --------------------------- 页面与常量 ---------------------------
st.set_page_config(page_title="专家抽样助手", page_icon="🎯", layout="wide")

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

# 可选字段，用于仿制界面更多筛选/展示
OPTIONAL_ALIASES: Dict[str, List[str]] = {
    "gender": ["性别", "Gender"],
    "birth": ["出生年月", "出生日期", "出生", "生日"],
    "idcard": ["身份证号", "身份证号码", "证件号码"],
    "org": ["供职单位", "单位", "工作单位", "所在单位"],
    "position": ["职务", "岗位", "职位"],
    "degree": ["学历", "最高学历"],
    # 删除注册资格筛选：不再在UI中使用
    # "qualification": ["注册资格", "资格", "注册证书"],
    "contact": ["联系方式", "联系电话", "联系手机", "手机号码", "手机号", "电话"],
    "contact_result": ["联系结果", "回访结果"],
    "reg_org": ["注册单位", "注册单位名称", "注册机构", "注册单位/机构"],
}

SUPPORTED_EXTENSIONS = [".xlsx", ".xls", ".csv"]


# --------------------------- 工具函数 ---------------------------
@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        # 自动尝试常见编码
        for enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename_lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="xlrd")
    else:
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df.columns = [str(c).strip() for c in new_df.columns]
    return new_df


def _is_empty_value(v: object) -> bool:
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def auto_locate_header_by_keyword(df: pd.DataFrame) -> pd.DataFrame:
    """根据包含“序号”等关键词的行自动定位表头。
    策略：行中包含“序号”，且与常见表头关键词交集>=3，则该行为表头；
    以该行之下一行为数据开始；将该行赋为列名。
    """
    if df.empty:
        return df
    s = df.astype(str).applymap(lambda x: str(x).strip())
    header_keywords = {"序号", "姓名", "性别", "出生年月", "身份证号", "专业", "手机号码", "手机号", "供职单位", "单位", "职称", "职务", "电子邮箱", "邮箱", "备注"}
    header_row_idx: Optional[int] = None
    for i, row in s.iterrows():
        vals = [v for v in row.tolist() if not _is_empty_value(v)]
        if not vals:
            continue
        if "序号" in vals:
            if len(set(vals) & header_keywords) >= 3:
                header_row_idx = i
                break
    if header_row_idx is None:
        return df
    columns = s.iloc[header_row_idx].tolist()
    columns = [c if not _is_empty_value(c) else f"列{j+1}" for j, c in enumerate(columns)]
    body = df.iloc[header_row_idx + 1 :].copy()
    body.columns = columns
    return normalize_columns(body)


def drop_group_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除仅包含一个非空文本，且该文本疑似分组标题的行，例如“信息化专业”。
    判定规则：
    - 行内非空单元格数量<=2，且
    - 存在非空值满足：等于“信息化专业”或以“专业”结尾或以“类别”结尾
    """
    if df.empty:
        return df
    s = df.astype(str).applymap(lambda x: str(x).strip())
    mask_drop = []
    for _, row in s.iterrows():
        vals = [v for v in row.tolist() if not _is_empty_value(v)]
        if len(vals) <= 2:
            candidate = "".join(vals)
            candidate = candidate.replace(" ", "")
            if candidate in {"信息化专业"} or candidate.endswith("专业") or candidate.endswith("类别"):
                mask_drop.append(True)
                continue
        mask_drop.append(False)
    return df.loc[[not x for x in mask_drop]]


def drop_embedded_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除表内重复的表头行（例如分页后又出现一次‘姓名/专业/职称/...’）。
    规则：如果本行含有至少3个字段名关键字，则判定为表头行。
    """
    if df.empty:
        return df
    header_keywords = set(REQUIRED_FIELDS_CN.values()) | {"性别", "出生年月", "身份证号", "单位", "供职单位", "职务", "电子邮箱", "备注"}
    s = df.astype(str).applymap(lambda x: str(x).strip())
    keep_flags: List[bool] = []
    for _, row in s.iterrows():
        row_set = set(v for v in row.tolist() if not _is_empty_value(v))
        intersect_cnt = len(row_set & header_keywords)
        keep_flags.append(intersect_cnt < 3)
    return df.loc[keep_flags]


def _find_first_match(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_lc = {c: str(c).lower() for c in cols}
    for a in aliases:
        # 精确
        for c in cols:
            if str(c).strip() == a:
                return c
        # 模糊包含
        a_lc = a.lower()
        for c, c_lc in cols_lc.items():
            if a_lc in c_lc:
                return c
    return None


def guess_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {k: None for k in REQUIRED_FIELDS_CN.keys()}

    for field, aliases in POSSIBLE_COLUMN_ALIASES.items():
        col = _find_first_match(df, aliases)
        if col is not None:
            mapping[field] = col
    return mapping


def build_mapped_df(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    selected_cols = {}
    for key, cn in REQUIRED_FIELDS_CN.items():
        col = mapping.get(key)
        if col is None:
            errors.append(f"未选择字段：{cn}")
        else:
            selected_cols[cn] = df[col]
    if errors:
        return pd.DataFrame(), errors

    result = pd.DataFrame(selected_cols)

    # 轻度清洗
    for c in result.columns:
        result[c] = result[c].astype(str).str.strip().replace({"nan": "", "None": "", "null": "", "NaN": ""})

    # 手机号仅保留数字
    result[REQUIRED_FIELDS_CN["phone"]] = (
        result[REQUIRED_FIELDS_CN["phone"]]
        .str.replace(r"[^0-9]", "", regex=True)
        .str.slice(0, 20)
    )

    # 删除疑似标题/分隔行（映射后再兜底一次）
    name_cn = REQUIRED_FIELDS_CN["name"]
    major_cn = REQUIRED_FIELDS_CN["major"]
    title_cn = REQUIRED_FIELDS_CN["title"]
    mask = ~(
        (result[name_cn].eq("") & (result[[major_cn, title_cn]].replace("", pd.NA).notna().sum(axis=1) <= 1))
        | (result[name_cn].isin(["姓名", "信息化专业"]))
        | (result[major_cn].isin(["信息化专业"]))
    )
    result = result.loc[mask]

    return result, []


def describe_dataframe(df: pd.DataFrame, mapped_df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    summary = {
        "行数": int(df.shape[0]),
        "列数": int(df.shape[1]),
        "列名": list(map(str, df.columns)),
    }
    if mapped_df is not None and not mapped_df.empty:
        major = REQUIRED_FIELDS_CN["major"]
        title = REQUIRED_FIELDS_CN["title"]
        summary.update(
            {
                f"唯一{major}数": int(mapped_df[major].replace("", pd.NA).nunique(dropna=True)),
                f"唯一{title}数": int(mapped_df[title].replace("", pd.NA).nunique(dropna=True)),
                "缺失统计": mapped_df.replace("", pd.NA).isna().sum().to_dict(),
            }
        )
    return summary


def compute_sample_size(total: int, mode: str, count: int, percent: float) -> int:
    if total <= 0:
        return 0
    if mode == "按数量":
        return int(max(0, min(total, count)))
    # 按百分比
    n = int(np.floor(total * (percent / 100.0)))
    return max(1 if total > 0 else 0, min(total, n))


def sample_rows(df: pd.DataFrame, sample_size: int, seed: Optional[int]) -> pd.DataFrame:
    if sample_size <= 0 or df.empty:
        return df.iloc[0:0]
    rng = np.random.default_rng(None if seed in (None, "") else int(seed))
    idx = rng.choice(df.index.values, size=sample_size, replace=False)
    return df.loc[idx]


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()


def detect_optional_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {}
    for k, aliases in OPTIONAL_ALIASES.items():
        found[k] = _find_first_match(df, aliases)
    return found


def compute_age_series(birth_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(birth_series, errors="coerce")
    today = pd.Timestamp.today().normalize()
    age = (today - dt).dt.days // 365
    return age


# --------------------------- 应用主体 ---------------------------
def main() -> None:
    st.title("🎯 专家抽样助手")
    st.caption("上传专家库（Excel/CSV），自动定位表头 → 清洗 → 映射 → 筛选/抽样 → 导出")

    uploaded = st.file_uploader(
        label="上传文件 (支持 .xlsx/.xls/.csv)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info("请先上传文件。支持 Excel 或 CSV。")
        st.stop()

    try:
        raw_df = load_dataframe(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        st.stop()

    # 自动定位表头行（基于“序号”等关键字）
    df = auto_locate_header_by_keyword(raw_df)
    # 规范列名 + 清洗杂项行
    df = normalize_columns(df)
    df = drop_group_header_rows(df)
    df = drop_embedded_header_rows(df)

    st.subheader("数据预览（自动定位表头 + 清洗后）")
    st.dataframe(df.head(50), use_container_width=True)

    # 字段映射（必填）
    st.subheader("字段映射")
    default_mapping = guess_mapping(df)

    mapping: Dict[str, Optional[str]] = {}
    cols = st.columns(5)
    for i, (key, cn) in enumerate(REQUIRED_FIELDS_CN.items()):
        with cols[i % 5]:
            mapping[key] = st.selectbox(
                label=f"{cn} 字段",
                options=[None] + list(df.columns),
                index=(
                    [None] + list(df.columns)
                ).index(default_mapping.get(key)) if default_mapping.get(key) in df.columns else 0,
                format_func=lambda x: "未选择" if x is None else str(x),
            )

    mapped_df, errors = build_mapped_df(df, mapping)
    if errors:
        st.warning("; ".join(errors))
        st.stop()

    optional_cols = detect_optional_columns(df)

    # 概要信息
    with st.expander("概要信息", expanded=False):
        summary = describe_dataframe(df, mapped_df)
        st.json(summary, expanded=False)

    # 两种模式：简洁模式、完整模式(1:1)
    tab_simple, tab_clone = st.tabs(["简洁模式", "完整模式(1:1)"])

    # ---------- 简洁模式（原有） ----------
    with tab_simple:
        st.subheader("筛选")
        col_left, col_right = st.columns(2)
        major_cn = REQUIRED_FIELDS_CN["major"]
        title_cn = REQUIRED_FIELDS_CN["title"]

        with col_left:
            majors = sorted([v for v in mapped_df[major_cn].dropna().unique() if str(v).strip() != ""])
            selected_majors = st.multiselect("按专业筛选（可多选，留空为全部）", majors)
        with col_right:
            titles = sorted([v for v in mapped_df[title_cn].dropna().unique() if str(v).strip() != ""])
            selected_titles = st.multiselect("按职称筛选（可多选，留空为全部）", titles)

        filtered_df = mapped_df.copy()
        if selected_majors:
            filtered_df = filtered_df[filtered_df[major_cn].isin(selected_majors)]
        if selected_titles:
            filtered_df = filtered_df[filtered_df[title_cn].isin(selected_titles)]

        st.caption(f"筛选后共有 {filtered_df.shape[0]} 条记录。")

        st.subheader("随机抽样")
        mode = st.radio("抽样方式", options=["按数量", "按百分比"], horizontal=True, key="simple_mode")
        col1, col2, col3 = st.columns(3)
        with col1:
            seed = st.text_input("随机种子（可选，留空则每次不同）", value="")
        with col2:
            count = st.number_input("抽取数量", min_value=0, value=10, step=1)
        with col3:
            percent = st.slider("抽取百分比", min_value=1, max_value=100, value=20, step=1)

        sample_size = compute_sample_size(
            total=filtered_df.shape[0], mode=mode, count=int(count), percent=float(percent)
        )
        st.caption(f"将抽取 {sample_size} 条记录。")

        if st.button("开始抽样", type="primary", key="simple_do_sample"):
            sampled = sample_rows(filtered_df, sample_size, seed if seed != "" else None)
            st.success(f"抽样完成，共 {sampled.shape[0]} 条。")
            st.dataframe(sampled, use_container_width=True)

            # 下载
            csv_bytes = sampled.to_csv(index=False).encode("utf-8-sig")
            excel_bytes = to_excel_bytes(sampled)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="下载 CSV",
                    data=csv_bytes,
                    file_name="抽样名单.csv",
                    mime="text/csv",
                )
            with dl_col2:
                st.download_button(
                    label="下载 Excel",
                    data=excel_bytes,
                    file_name="抽样名单.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ),
                )

    # ---------- 仿制模式(1:1) ----------
    with tab_clone:
        st.subheader("项目信息")
        st.caption("星号为必填项。仅用于记录显示，不影响抽样逻辑。")
        col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 1])
        with col_a:
            proj_name = st.text_input("项目名称", value="", placeholder="例如：XX项目采购")
        with col_b:
            owner_name = st.text_input("*招标人名称", value="", placeholder="例如：XX单位")
        with col_c:
            bid_no = st.text_input("*招标编号", value="", placeholder="例如：SD-2025-001")
        with col_d:
            draw_time = st.text_input("*抽取时间", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.subheader("抽取条件")
        c1, c2, c3, c4 = st.columns(4)
        major_cn = REQUIRED_FIELDS_CN["major"]
        title_cn = REQUIRED_FIELDS_CN["title"]

        with c1:
            major_opts = sorted([v for v in df[major_cn].dropna().unique() if str(v).strip() != ""]) if major_cn in df.columns else []
            major_sel = st.multiselect("专业", major_opts, key="clone_major")
        with c2:
            title_opts = sorted([v for v in df[title_cn].dropna().unique() if str(v).strip() != ""]) if title_cn in df.columns else []
            title_sel = st.multiselect("职称", title_opts, key="clone_title")
        with c3:
            gender_col = optional_cols.get("gender")
            gender_opts = sorted([v for v in df[gender_col].dropna().unique() if str(v).strip() != ""]) if gender_col else []
            gender_sel = st.selectbox("性别", ["全部"] + gender_opts if gender_opts else ["全部"], index=0)
        with c4:
            birth_col = optional_cols.get("birth")
            if birth_col:
                ages = compute_age_series(df[birth_col])
                min_age = int(ages.min()) if ages.notna().any() else 18
                max_age = int(ages.max()) if ages.notna().any() else 80
                age_range = st.slider("年龄", min_value=max(16, min_age), max_value=min(100, max_age), value=(max(18, min_age), min(65, max_age)))
            else:
                st.write("年龄：无出生年月，无法筛选")
                age_range = None

        c5, c6 = st.columns(2)
        with c5:
            degree_col = optional_cols.get("degree")
            degree_opts = sorted([v for v in df[degree_col].dropna().unique() if str(v).strip() != ""]) if degree_col else []
            degree_sel = st.selectbox("学历", ["全部"] + degree_opts if degree_opts else ["全部"], index=0)
        with c6:
            num_people = st.number_input("人数", min_value=0, value=10, step=1, help="输入需要抽取的人数")

        st.subheader("回避（多选剔除）")
        org_col = optional_cols.get("org")
        reg_org_col = optional_cols.get("reg_org")
        r1, r2 = st.columns(2)
        with r1:
            avoid_orgs = st.multiselect(
                "回避工作单位",
                sorted([v for v in df[org_col].dropna().unique() if str(v).strip() != ""]) if org_col else [],
                help="选择这些单位的专家将被剔除")
        with r2:
            avoid_reg_orgs = st.multiselect(
                "回避注册单位",
                sorted([v for v in df[reg_org_col].dropna().unique() if str(v).strip() != ""]) if reg_org_col else [],
                help="选择这些注册单位的专家将被剔除")

        # 基于条件过滤
        clone_filtered = df.copy()
        if major_cn in clone_filtered.columns and major_sel:
            clone_filtered = clone_filtered[clone_filtered[major_cn].isin(major_sel)]
        if title_cn in clone_filtered.columns and title_sel:
            clone_filtered = clone_filtered[clone_filtered[title_cn].isin(title_sel)]
        if gender_col and gender_sel != "全部":
            clone_filtered = clone_filtered[clone_filtered[gender_col] == gender_sel]
        if birth_col and age_range is not None:
            ages = compute_age_series(clone_filtered[birth_col])
            clone_filtered = clone_filtered[(ages >= age_range[0]) & (ages <= age_range[1])]
        if degree_col and degree_sel != "全部":
            clone_filtered = clone_filtered[clone_filtered[degree_col] == degree_sel]
        # 回避单位剔除
        if org_col and avoid_orgs:
            clone_filtered = clone_filtered[~clone_filtered[org_col].isin(avoid_orgs)]
        if reg_org_col and avoid_reg_orgs:
            clone_filtered = clone_filtered[~clone_filtered[reg_org_col].isin(avoid_reg_orgs)]

        st.caption(f"符合条件记录：{clone_filtered.shape[0]} 人")

        # 操作按钮（更贴合非技术用户）
        st.markdown("---")
        btn_col1, btn_col2 = st.columns([1, 1])
        do_draw = btn_col1.button("🎯 抽取", type="primary", use_container_width=True, key="clone_draw")
        do_print = btn_col2.button("🖨️ 打印列表", use_container_width=True, key="clone_print")

        selected = pd.DataFrame()
        if do_draw:
            selected = sample_rows(clone_filtered, int(num_people), seed=None)
            st.success(f"已抽取 {selected.shape[0]} 人。")
            # 展示与下载（使用基本五字段，如有可选字段则追加展示）
            base_cols = [
                REQUIRED_FIELDS_CN["name"],
                major_cn if major_cn in selected.columns else None,
                REQUIRED_FIELDS_CN["phone"],
                title_cn if title_cn in selected.columns else None,
                REQUIRED_FIELDS_CN["email"],
            ]
            base_cols = [c for c in base_cols if c and c in selected.columns]
            show_cols = base_cols.copy()
            for opt_key in ["gender", "birth", "org", "position", "reg_org"]:
                col = optional_cols.get(opt_key)
                if col and col in selected.columns:
                    show_cols.append(col)
            st.dataframe(selected[show_cols] if show_cols else selected, use_container_width=True)

            # 下载
            csv_bytes = (selected[show_cols] if show_cols else selected).to_csv(index=False).encode("utf-8-sig")
            excel_bytes = to_excel_bytes(selected[show_cols] if show_cols else selected)
            st.download_button("下载 CSV", data=csv_bytes, file_name="抽取结果.csv", mime="text/csv")
            st.download_button("下载 Excel", data=excel_bytes, file_name="抽取结果.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # 打印（生成简洁HTML并触发浏览器打印）
        if do_print:
            preview = clone_filtered.head(200)
            html_table = preview.to_html(index=False)
            st_html(
                f"""
                <div>
                  <style>
                    table {{border-collapse: collapse; width: 100%; font-size: 12px;}}
                    th, td {{border: 1px solid #999; padding: 6px;}}
                  </style>
                  <h3 style='margin:8px 0'>项目名称：{proj_name or ''} ｜ 招标人：{owner_name or ''} ｜ 招标编号：{bid_no or ''} ｜ 抽取时间：{draw_time or ''}</h3>
                  {html_table}
                  <script>window.print();</script>
                </div>
                """,
                height=520,
            )


if __name__ == "__main__":
    main()
