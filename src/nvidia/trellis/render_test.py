"""
Test script for rendering GLB to PNG.

Usage:
  uv run python -m nvidia.trellis.render_test
"""

from pathlib import Path


def render_with_pyvista(glb_path: Path, output_path: Path) -> bool:
    """Try rendering with pyvista (VTK-based)."""
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        pv.start_xvfb()  # Try to start virtual framebuffer

        mesh = pv.read(str(glb_path))

        plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
        plotter.add_mesh(mesh)
        plotter.camera_position = "iso"
        plotter.screenshot(str(output_path))
        plotter.close()

        return output_path.exists()
    except Exception as e:
        print(f"pyvista failed: {e}")
        return False


def render_with_trimesh_pyrender(glb_path: Path, output_path: Path) -> bool:
    """Try rendering with trimesh + pyrender."""
    try:
        import os

        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

        import numpy as np
        import pyrender
        import trimesh
        from PIL import Image

        mesh = trimesh.load(glb_path)

        scene = pyrender.Scene()

        if isinstance(mesh, trimesh.Scene):
            for name, geom in mesh.geometry.items():
                if hasattr(geom, "vertices"):
                    pymesh = pyrender.Mesh.from_trimesh(geom)
                    scene.add(pymesh)
        else:
            pymesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(pymesh)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        scene.add(camera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(400, 400)
        color, depth = r.render(scene)
        r.delete()

        img = Image.fromarray(color)
        img.save(output_path)

        return output_path.exists()
    except Exception as e:
        print(f"trimesh+pyrender failed: {e}")
        return False


def render_with_blender(glb_path: Path, output_path: Path) -> bool:
    """Try rendering with Blender (if installed)."""
    import shutil
    import subprocess

    blender = shutil.which("blender")
    if not blender:
        print("Blender not found in PATH")
        return False

    script = f"""
import bpy
import sys

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB
bpy.ops.import_scene.gltf(filepath="{glb_path}")

# Add camera
bpy.ops.object.camera_add(location=(2, -2, 2))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = camera

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))

# Render settings
bpy.context.scene.render.resolution_x = 400
bpy.context.scene.render.resolution_y = 400
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.filepath = "{output_path}"

# Render
bpy.ops.render.render(write_still=True)
"""

    try:
        result = subprocess.run(
            [blender, "--background", "--python-expr", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return output_path.exists()
    except Exception as e:
        print(f"Blender failed: {e}")
        return False


def render_with_subprocess_pyvista(glb_path: Path, output_path: Path) -> bool:
    """Run pyvista in a subprocess to isolate OpenGL context."""
    import subprocess
    import sys

    script = f"""
import pyvista as pv
pv.OFF_SCREEN = True

mesh = pv.read("{glb_path}")
plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
plotter.add_mesh(mesh)
plotter.camera_position = "iso"
plotter.screenshot("{output_path}")
plotter.close()
print("OK")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={**__import__("os").environ, "DISPLAY": ""},  # Disable display
        )
        print(f"Subprocess stdout: {result.stdout}")
        print(f"Subprocess stderr: {result.stderr[:500] if result.stderr else '(none)'}")
        return output_path.exists() and "OK" in result.stdout
    except Exception as e:
        print(f"Subprocess pyvista failed: {e}")
        return False


def main():
    glb_path = Path("src/nvidia/trellis/output/Cylindrical_white_ceramic_base.glb")
    if not glb_path.exists():
        # Try another file
        glb_path = Path("src/nvidia/trellis/output/text.glb")

    if not glb_path.exists():
        print(f"No GLB file found")
        return 1

    print(f"Testing GLB rendering: {glb_path}")
    print(f"File size: {glb_path.stat().st_size / 1024:.1f} KB")
    print()

    output_dir = Path("/tmp/glb_render_test")
    output_dir.mkdir(exist_ok=True)

    methods = [
        ("subprocess_pyvista", render_with_subprocess_pyvista),
        ("pyvista", render_with_pyvista),
        ("trimesh_pyrender", render_with_trimesh_pyrender),
        ("blender", render_with_blender),
    ]

    for name, func in methods:
        output_path = output_dir / f"{name}.png"
        if output_path.exists():
            output_path.unlink()

        print(f"Testing {name}...")
        success = func(glb_path, output_path)

        if success:
            print(f"  SUCCESS: {output_path} ({output_path.stat().st_size} bytes)")
        else:
            print(f"  FAILED")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
