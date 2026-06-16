# Live Objects - Mesh Directory

Place 3D meshes here for use with the live tracker.

## Directory Structure

```
local_data/live_objects/
    <object-label>/
        <any_name>.ply or .obj
```

- **Directory name = object label** -- this must match the `--object-label` argument
  passed to the tracker script.
- Place exactly one `.ply` or `.obj` mesh file per label directory.
- Mesh units must be in **millimeters** (the pipeline converts to meters internally).

## Example

```
local_data/live_objects/
    puzzle-half-trapezoid/
        puzzle_half_trapezoid.ply
    coffee-mug/
        mug_model.obj
```

Then run:
```bash
python -m megapose.scripts.run_live_tracker --object-label puzzle-half-trapezoid
```

## Mesh Requirements

- Format: `.ply` or `.obj` (loaded via trimesh)
- Units: millimeters
- The mesh should be centered and oriented consistently with how YOLO
  will observe the object
- Watertight meshes are preferred but not strictly required

## Adding a New Object

1. Create a directory: `local_data/live_objects/<my-object>/`
2. Place your mesh file (`.ply` or `.obj`) inside it
3. Run the tracker with `--object-label my-object`
4. For best results, also train a custom YOLO model (see `yolo/training/README.md`)
