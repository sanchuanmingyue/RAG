"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import streamlit as st


def apply_app_style() -> None:
    """Apply one lightweight visual system across all Streamlit pages."""

    st.markdown(
        """
        <style>
        :root {
            --rag-page-bg: #F7F7F8;
            --rag-sidebar: #F3F3F5;
            --rag-card: #FFFFFF;
            --rag-elevated-card: #FAFAFB;
            --rag-border: #E6E6E8;
            --rag-text: #202124;
            --rag-text-secondary: #71717A;
            --rag-accent: #6366F1;
            --rag-accent-hover: #5558E8;
            --rag-accent-soft: rgba(99, 102, 241, .10);
            --rag-primary: var(--rag-accent);
            --rag-primary-dark: var(--rag-accent-hover);
            --rag-primary-soft: var(--rag-accent-soft);
            --rag-ink: var(--rag-text);
            --rag-muted: var(--rag-text-secondary);
            --rag-line: var(--rag-border);
            --rag-panel: var(--rag-card);
            --rag-canvas: var(--rag-page-bg);
        }
        .stApp {background: var(--rag-canvas); color: var(--rag-ink);}
        .block-container {
            max-width: 1920px; padding: 4.35rem 2rem 2rem;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 247, 248, .96); border-bottom: 1px solid var(--rag-line);
        }
        [data-testid="stSidebar"] {
            background: var(--rag-sidebar); border-right: 1px solid var(--rag-line);
            width: 296px !important; min-width: 296px !important;
        }
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebarContent"] {padding-top: .55rem;}
        [data-testid="stSidebarContent"] .stButton > button {min-height: 2.35rem;}
        [data-testid="stMetric"] {
            background: var(--rag-card); border: 1px solid var(--rag-border); border-radius: 14px;
            padding: .75rem 1rem; box-shadow: none;
        }
        .stButton > button {border-radius: 9px; font-weight: 600;}
        .stButton > button[kind="primary"] {
            background: var(--rag-primary); border-color: var(--rag-primary); color: white;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--rag-primary-dark); border-color: var(--rag-primary-dark);
        }
        .st-key-top_navigation_bar {
            position: sticky; top: 3.75rem; z-index: 990; background: rgba(247, 247, 248, .97);
            backdrop-filter: blur(12px); padding: .4rem .15rem .7rem;
            border-bottom: 1px solid var(--rag-line); margin-bottom: .8rem; overflow: visible;
        }
        .st-key-top_navigation_bar .stButton > button {
            min-height: 2.45rem; border-radius: 8px; box-shadow: none;
            line-height: 1.2; overflow: visible; border-color: transparent;
            background: transparent; color: var(--rag-text-secondary);
        }
        .st-key-top_navigation_bar .stButton > button:hover {
            background: var(--rag-elevated-card); border-color: var(--rag-border); color: var(--rag-ink);
        }
        .st-key-top_navigation_bar .stButton > button[kind="primary"] {
            background: var(--rag-primary-soft); border-color: var(--rag-accent); color: var(--rag-accent);
        }
        .st-key-top_navigation_bar .stButton > button p {margin: 0; line-height: 1.2;}
        div[data-testid="stExpander"] {background: var(--rag-card); border-radius: 12px; border-color: var(--rag-border);}
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--rag-card); border: 1.5px dashed var(--rag-accent); border-radius: 12px;
        }
        div[data-testid="stFileUploaderDropzone"] button {color: var(--rag-primary-dark);}
        textarea:focus, input:focus {border-color: var(--rag-primary) !important;}
        .rag-brand {font-size: 1.12rem; font-weight: 780; color: var(--rag-text); white-space: nowrap; letter-spacing: -.01em;}
        .rag-brand span {color: var(--rag-primary); font-size: 1.25rem; margin-right: .35rem;}
        .rag-sidebar-title {font-size: 1.35rem; font-weight: 780; margin: .2rem 0 .15rem;}
        .rag-sidebar-section {
            margin: 1rem 0 .4rem; color: var(--rag-text-secondary); font-size: .73rem;
            font-weight: 750; letter-spacing: .08em; text-transform: uppercase;
        }
        .rag-file-status {
            background: var(--rag-primary-soft); border: 1px solid var(--rag-accent); border-radius: 10px;
            padding: .65rem .8rem; margin: .6rem 0; color: var(--rag-text);
        }
        .rag-file-status span {font-size: .8rem; color: var(--rag-text-secondary);}
        .rag-file-row {
            display: flex; gap: .55rem; align-items: flex-start; padding: .52rem .55rem;
            margin: .18rem 0; border-radius: 8px; color: var(--rag-text-secondary); overflow-wrap: anywhere;
        }
        .rag-file-row > span {color: var(--rag-text-secondary);}
        .rag-file-row small {display: block; color: var(--rag-text-secondary); margin-top: .15rem;}
        .rag-file-row.active {background: var(--rag-primary-soft); color: var(--rag-accent);}
        .rag-file-row.active > span {color: var(--rag-primary);}
        .rag-view-heading {margin: .2rem 0 1rem;}
        .rag-view-heading h2 {font-size: 1.45rem; margin: 0 0 .2rem;}
        .rag-view-heading p {margin: 0; color: var(--rag-text-secondary);}
        .rag-message-label {font-size: .76rem; margin: .35rem .3rem .22rem; color: var(--rag-text-secondary);}
        .rag-message-label.user {text-align: right; color: var(--rag-primary);}
        [class*="st-key-chat_user_"] {
            background: var(--rag-primary-soft); border-color: var(--rag-accent) !important; border-radius: 14px;
        }
        [class*="st-key-chat_assistant_"] {
            background: var(--rag-card); border-color: var(--rag-border) !important; border-radius: 14px;
        }
        .st-key-chat_history {
            background: var(--rag-card); border-color: var(--rag-line) !important;
            border-radius: 16px !important;
        }
        .rag-workspace-bar {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: .72rem .9rem; margin: 0 0 .75rem; background: var(--rag-card);
            border: 1px solid var(--rag-line); border-radius: 12px;
        }
        .rag-workspace-title {
            min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            font-size: .92rem; font-weight: 720; color: var(--rag-text);
        }
        .rag-workspace-meta {
            flex: 0 0 auto; font-size: .78rem; color: var(--rag-muted); text-align: right;
        }
        .rag-info-title {font-size: 1rem; font-weight: 760; margin: .1rem 0;}
        .rag-info-subtitle {font-size: .78rem; color: var(--rag-muted); margin-bottom: .65rem;}
        .rag-evidence-heading {font-size: .86rem; font-weight: 720; color: var(--rag-text); margin-bottom: .1rem;}
        .rag-score-row {display: flex; flex-wrap: wrap; gap: .35rem; margin: .35rem 0 .55rem;}
        .rag-score-pill {
            display: inline-block; padding: .18rem .46rem; border-radius: 999px;
            background: var(--rag-elevated-card); color: var(--rag-text-secondary); font-size: .69rem;
            border: 1px solid var(--rag-border);
        }
        [class*="st-key-evidence_card_"] {
            background: var(--rag-elevated-card); border-color: var(--rag-line) !important;
            border-radius: 12px !important;
        }
        .st-key-information_panel {
            background: var(--rag-card); border-color: var(--rag-line) !important;
            border-radius: 16px !important;
        }
        .st-key-workspace_question textarea {
            border-radius: 12px; background: var(--rag-card); min-height: 76px;
        }
        .st-key-multimodal_chat_enabled label p {white-space: nowrap;}
        .rag-hero {
            padding: 1.15rem 1.35rem; margin-bottom: 1rem; border-radius: 16px;
            color: var(--rag-text); background: var(--rag-elevated-card);
            border: 1px solid var(--rag-border);
        }
        .rag-hero h1 {font-size: 1.75rem; margin: 0 0 .35rem 0;}
        .rag-hero p {margin: 0; color: var(--rag-text-secondary);}
        .rag-badge {
            display: inline-block; padding: .18rem .55rem; margin-right: .35rem;
            border-radius: 999px; background: var(--rag-accent-soft); color: var(--rag-accent); font-size: .78rem;
        }
        @media (max-width: 1180px) {
            .block-container {padding-left: 1rem; padding-right: 1rem;}
            [data-testid="stSidebar"] {width: 276px !important; min-width: 276px !important;}
            .rag-workspace-meta {display: none;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_navigation(active: str = "Chat", *, on_main_page: bool = True) -> str:
    """Render the shared app navigation and switch between Streamlit pages safely."""

    main_destinations = {"Chat", "Search", "Files", "Tools"}
    if active not in main_destinations | {"Multimodal"}:
        active = "Chat"

    with st.container(key="top_navigation_bar"):
        brand_col, navigation_col = st.columns([1.15, 4], vertical_alignment="center")
        with brand_col:
            st.markdown('<div class="rag-brand"><span>◉</span> PaperReader</div>', unsafe_allow_html=True)
        with navigation_col:
            nav_columns = st.columns(4, gap="small")
            destinations = ("Chat", "Files", "Search", "Tools")
            for column, label in zip(nav_columns, destinations):
                if column.button(
                    label,
                    key=f"global_nav_{label.lower()}",
                    type="primary" if active == label else "secondary",
                    width="stretch",
                ):
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
