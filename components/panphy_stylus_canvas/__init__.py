import streamlit.components.v1 as components
from pathlib import Path
from typing import Optional

# Declare the component. Streamlit will serve index.html from this folder.
_component = components.declare_component(
    name="panphy_stylus_canvas",
    path=str(Path(__file__).parent),
)

def stylus_canvas(
    *,
    height: int,
    width: Optional[int] = None,
    stroke_width: int = 2,
    stroke_color: str = "#000000",
    background_color: str = "#ffffff",
    pen_only: bool = False,
    tool: str = "pen",  # 'pen' | 'eraser'
    command: Optional[str] = None,  # 'clear' | 'undo' | None
    command_nonce: int = 0,
    initial_data_url: Optional[str] = None,
    # Resizing UX
    resizable: bool = True,
    min_height: int = 220,
    max_height: int = 1200,
    storage_key: Optional[str] = None,  # localStorage key for persisting user-resized height
    key: Optional[str] = None,
):
    """PanPhy Stylus Canvas (custom Streamlit component).

    Returns a dict like:
      { "data_url": "data:image/png;base64,...", "is_empty": bool }

    Notes:
    - If width is None, the component will expand to the container width.
    - The canvas area is user-resizable vertically (like a text area) when resizable=True.
    """

    kwargs = dict(
        height=int(height),
        stroke_width=int(stroke_width),
        stroke_color=str(stroke_color),
        background_color=str(background_color),
        pen_only=bool(pen_only),
        tool=str(tool),
        command=command,
        command_nonce=int(command_nonce),
        initial_data_url=initial_data_url,
        resizable=bool(resizable),
        min_height=int(min_height),
        max_height=int(max_height),
        storage_key=storage_key,
        key=key,
        default=None,
    )
    # Only set width if explicitly requested, otherwise let Streamlit use container width.
    if width is not None:
        kwargs["width"] = int(width)

    return _component(**kwargs)
