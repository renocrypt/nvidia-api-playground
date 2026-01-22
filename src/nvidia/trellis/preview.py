"""
GLB preview utilities for terminal display.

Renders GLB files to 2D images for display in the terminal,
with fallback to system viewer if rendering fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def render_glb_to_image(glb_path: Path, output_path: Path | None = None) -> Path | None:
    """
    Render GLB to PNG image for terminal display.

    Runs in a subprocess to isolate OpenGL context from the TUI.
    Uses trimesh to extract textures and pyvista to render.

    Args:
        glb_path: Path to the GLB file
        output_path: Optional output path for the PNG. If None, uses temp file.

    Returns:
        Path to the rendered PNG, or None if rendering failed.
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".png"))

    # Use trimesh to load textures, pyvista to render (always offscreen)
    script = f'''
import os
import sys

# Force offscreen rendering before importing pyvista/vtk
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
# Disable display on Linux
os.environ.pop("DISPLAY", None)

import trimesh
import pyvista as pv
import numpy as np
from pathlib import Path
import tempfile

# Ensure offscreen mode
pv.OFF_SCREEN = True

glb_path = "{glb_path}"
output_path = "{output_path}"

try:
    # Load with trimesh to get textures
    scene = trimesh.load(glb_path)

    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))

    if isinstance(scene, trimesh.Scene):
        geometries = scene.geometry.items()
    else:
        geometries = [("mesh", scene)]

    for name, geom in geometries:
        if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
            continue

        # Convert trimesh to pyvista
        faces = np.hstack([[3] + list(f) for f in geom.faces])
        mesh = pv.PolyData(geom.vertices, faces)

        # Try to get texture
        has_texture = False
        if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
            mat = geom.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                # Save texture to temp file
                tex_path = Path(tempfile.mktemp(suffix=".png"))
                mat.baseColorTexture.save(tex_path)

                # Get UV coordinates
                if hasattr(geom.visual, "uv") and geom.visual.uv is not None:
                    mesh.active_texture_coordinates = geom.visual.uv
                    tex = pv.read_texture(str(tex_path))
                    plotter.add_mesh(mesh, texture=tex)
                    has_texture = True

        if not has_texture:
            # Use vertex colors if available
            if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
                colors = geom.visual.vertex_colors
                if colors is not None and len(colors) == len(geom.vertices):
                    mesh["colors"] = colors[:, :3]  # RGB only
                    plotter.add_mesh(mesh, scalars="colors", rgb=True)
                    has_texture = True

        if not has_texture:
            plotter.add_mesh(mesh, color="lightgray")

    plotter.camera_position = "iso"
    plotter.screenshot(output_path)
    plotter.close()
    print("OK")

except Exception as e:
    print(f"FAIL: {{e}}")
    import traceback
    traceback.print_exc()
    exit(1)
'''

    try:
        # Run with DISPLAY unset to avoid X11 issues
        env = os.environ.copy()
        env.pop("DISPLAY", None)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode == 0 and "OK" in result.stdout and output_path.exists():
            return output_path
    except Exception:
        pass

    return None


def open_in_viewer(glb_path: Path) -> bool:
    """
    Open GLB file in system default viewer.

    Args:
        glb_path: Path to the GLB file

    Returns:
        True if viewer was launched, False otherwise.
    """
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(glb_path)])
        elif sys.platform == "win32":
            subprocess.Popen(["start", "", str(glb_path)], shell=True)
        else:
            subprocess.Popen(["xdg-open", str(glb_path)])
        return True
    except Exception:
        return False


def get_glb_info(glb_path: Path) -> dict | None:
    """
    Get basic info about a GLB file.

    Args:
        glb_path: Path to the GLB file

    Returns:
        Dict with vertices, faces, file_size, or None if failed.
    """
    try:
        import trimesh
    except ImportError:
        return {"file_size": glb_path.stat().st_size}

    try:
        mesh = trimesh.load(glb_path)

        info = {"file_size": glb_path.stat().st_size}

        if isinstance(mesh, trimesh.Scene):
            # Scene with multiple geometries
            total_vertices = 0
            total_faces = 0
            for geom in mesh.geometry.values():
                if hasattr(geom, "vertices"):
                    total_vertices += len(geom.vertices)
                if hasattr(geom, "faces"):
                    total_faces += len(geom.faces)
            info["vertices"] = total_vertices
            info["faces"] = total_faces
            info["geometries"] = len(mesh.geometry)
        elif hasattr(mesh, "vertices"):
            info["vertices"] = len(mesh.vertices)
            if hasattr(mesh, "faces"):
                info["faces"] = len(mesh.faces)

        return info
    except Exception:
        return {"file_size": glb_path.stat().st_size}
