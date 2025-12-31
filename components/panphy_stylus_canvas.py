"""
PanPhy Stylus Canvas: minimal Streamlit custom component wrapper.

Frontend: components/panphy_stylus_canvas_frontend/index.html
The frontend implements Streamlit component postMessage plumbing and returns a JSON dict:
{
  "data_url": "data:image/png;base64,...",
  "is_empty": bool,
  "w": int,
  "h": int
}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "panphy_stylus_canvas_frontend"
_component = components.declare_component("panphy_stylus_canvas", path=str(_FRONTEND_DIR))


def stylus_canvas(
    *,
    stroke_width: int = 2,
    stroke_color: str = "#000000",
    background_color: str = "#F0F2F6",
    height: int = 400,
    width: int = 600,
    pen_only: bool = True,
    tool: str = "pen",
    command: Optional[str] = None,
    command_nonce: int = 0,
    initial_data_url: Optional[str] = None,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Render the canvas component and return the latest JSON value."""
    return _component(
        stroke_width=int(stroke_width),
        stroke_color=str(stroke_color),
        background_color=str(background_color),
        height=int(height),
        width=int(width),
        pen_only=bool(pen_only),
        tool=str(tool),
        command=command,
        command_nonce=int(command_nonce),
        initial_data_url=initial_data_url,
        default={"data_url": None, "is_empty": True, "w": int(width), "h": int(height)},
        key=key,
    )
