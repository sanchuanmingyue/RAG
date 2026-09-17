"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import streamlit as st


def apply_app_style() -> None:
    """Apply one lightweight visual system across all Streamlit pages."""

    st.markdown(
        """
        <style>
        :root {--rag-green: #10b981; --rag-green-dark: #059669; --rag-mint: #e9fbf4;}
        .stApp {background: #fbfcfd; color: #252a31;}
        .block-container {max-width: 1800px; padding-top: 4.75rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {
            background: #f7f8fa; border-right: 1px solid #e7eaee; min-width: 318px;
        }
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebarContent"] {padding-top: .7rem;}
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e6eaf0; border-radius: 14px;
            padding: .75rem 1rem; box-shadow: 0 4px 16px rgba(15, 23, 42, .04);
        }
        .stButton > button {border-radius: 9px; font-weight: 600;}
        .stButton > button[kind="primary"] {
            background: var(--rag-green); border-color: var(--rag-green); color: white;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--rag-green-dark); border-color: var(--rag-green-dark);
        }
        .st-key-top_navigation_bar {
            position: relative; z-index: 1; background: #fbfcfd;
            padding: .55rem .15rem .7rem; border-bottom: 1px solid #e8ebef;
            margin-bottom: .9rem; overflow: visible;
        }
        .st-key-top_navigation_bar .stButton > button {
            min-height: 2.55rem; border-radius: 7px; box-shadow: none;
            line-height: 1.2; overflow: visible;
        }
        .st-key-top_navigation_bar .stButton > button p {margin: 0; line-height: 1.2;}
        div[data-testid="stExpander"] {border-radius: 12px; border-color: #e3e8ef;}
        div[data-testid="stFileUploaderDropzone"] {
            background: white; border: 1.5px dashed #9edbc5; border-radius: 12px;
        }
        div[data-testid="stFileUploaderDropzone"] button {color: var(--rag-green-dark);}
        textarea:focus, input:focus {border-color: var(--rag-green) !important;}
        .rag-brand {font-size: 1.12rem; font-weight: 750; color: #242a31; white-space: nowrap;}
        .rag-brand span {color: var(--rag-green); font-size: 1.25rem; margin-right: .35rem;}
        .rag-sidebar-title {font-size: 1.28rem; font-weight: 750; margin: .25rem 0 .15rem;}
        .rag-file-status {
            background: var(--rag-mint); border: 1px solid #c7efdf; border-radius: 10px;
            padding: .65rem .8rem; margin: .6rem 0; color: #275746;
        }
        .rag-file-status span {font-size: .8rem; color: #4b7567;}
        .rag-file-row {
            display: flex; gap: .55rem; align-items: flex-start; padding: .52rem .55rem;
            margin: .18rem 0; border-radius: 8px; color: #535961; overflow-wrap: anywhere;
        }
        .rag-file-row > span {color: #b6bdc5;}
        .rag-file-row small {display: block; color: #9198a1; margin-top: .15rem;}
        .rag-file-row.active {background: var(--rag-mint); color: #1f5e48;}
        .rag-file-row.active > span {color: var(--rag-green);}
        .rag-view-heading {margin: .2rem 0 1rem;}
        .rag-view-heading h2 {font-size: 1.45rem; margin: 0 0 .2rem;}
        .rag-view-heading p {margin: 0; color: #747b85;}
        .rag-message-label {font-size: .76rem; margin: .35rem .3rem .22rem; color: #858c95;}
        .rag-message-label.user {text-align: right; color: #258365;}
        [class*="st-key-chat_user_"] {
            background: #e9fbf4; border-color: #c7efdf !important; border-radius: 14px;
        }
        [class*="st-key-chat_assistant_"] {
            background: #ffffff; border-color: #e4e8ec !important; border-radius: 14px;
        }
        .rag-hero {
            padding: 1.15rem 1.35rem; margin-bottom: 1rem; border-radius: 16px;
            color: #172033; background: linear-gradient(135deg, #edfbf6 0%, #f7fbf9 100%);
            border: 1px solid #d4eee4;
        }
        .rag-hero h1 {font-size: 1.75rem; margin: 0 0 .35rem 0;}
        .rag-hero p {margin: 0; color: #526078;}
        .rag-badge {
            display: inline-block; padding: .18rem .55rem; margin-right: .35rem;
            border-radius: 999px; background: #dcf7ed; color: #167657; font-size: .78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_navigation(active: str = "Chat", *, on_main_page: bool = True) -> str:
    """Render the shared app navigation and switch between Streamlit pages safely."""

    main_destinations = {"Chat", "Files", "Tools"}
    if active not in main_destinations | {"Multimodal"}:
        active = "Chat"

    with st.container(key="top_navigation_bar"):
        brand_col, navigation_col = st.columns([1.05, 5], vertical_alignment="center")
        with brand_col:
            st.markdown('<div class="rag-brand"><span>◉</span> PaperReader</div>', unsafe_allow_html=True)
        with navigation_col:
            nav_columns = st.columns([1, 1, 1, 1.45, 3.3], gap="small")
            destinations = ("Chat", "Files", "Tools", "Multimodal")
            for column, label in zip(nav_columns[:4], destinations):
                if column.button(
                    label,
                    key=f"global_nav_{label.lower()}",
                    type="primary" if active == label else "secondary",
                    width="stretch",
                ):
                    if label == "Multimodal":
                        st.switch_page("pages/2_多模态论文阅读.py")
                    st.session_state.main_navigation = label
                    if not on_main_page:
                        st.switch_page("app.py")
                    st.rerun()
    return st.session_state.get("main_navigation", active)


def render_page_header(title: str, description: str, badges: list[str] | None = None) -> None:
    badge_html = "".join(f'<span class="rag-badge">{badge}</span>' for badge in (badges or []))
    st.markdown(
        f'<section class="rag-hero"><h1>{title}</h1><p>{description}</p>'
        f'<div style="margin-top:.65rem">{badge_html}</div></section>',
        unsafe_allow_html=True,
    )
