"""Convert our physical designs into FairShip SST geometry configs.

Our designs (``config/design/*.yaml``) are ``{stations: [z(cm) x4], angle: rad}``. FairShip's SST
schema (``FairShip/geometry/strawtubes_config.yaml``) holds the stereo angle as ``view_angle`` in
DEGREES; the patched ``shipDet_conf.py`` also reads a TOP-LEVEL ``stations:`` list and feeds it to
``strawtubes.SetzPositions`` (falling back to ``geometry_config.py``'s ``TrackStation{1..4}.z`` when
absent). So BOTH design dofs fit this file: we start from the template, override only ``view_angle``
per design, and append the design's 4 station-centre z's as the top-level ``stations`` key -- every
other SST field stays byte-for-byte (same format/comments). Output: one
``strawtubes_config_<name>.yaml`` per design.

    python scripts/design_to_strawtubes.py
"""

import glob
import math
import os

import yaml

TEMPLATE = "FairShip/geometry/strawtubes_config.yaml"
OUT_DIR = "FairShip/geometry"  # one strawtubes_config_<design>.yaml per design, beside the stock file


def convert(design_path, template_lines):
    """Return (converted SST yaml text, stations, view_angle_deg) for one design."""
    design = yaml.safe_load(open(design_path))
    view_angle_deg = math.degrees(float(design["angle"]))
    name = os.path.basename(design_path)
    stations = [float(z) for z in design["stations"]]
    out = []
    for line in template_lines:
        if line.lstrip().startswith("view_angle:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}view_angle: {view_angle_deg:.6g} # Stereo angle in degree (from {name})\n")
        else:
            out.append(line)
    # Top-level (sibling of SST) station-centre z's -> strawtubes.SetzPositions in shipDet_conf.py.
    if not out[-1].endswith("\n"):
        out.append("\n")
    out.append(f"\n# Station-centre z in cm -> strawtubes.SetzPositions (from {name})\n")
    out.append("stations: [" + ", ".join(f"{z:.4f}" for z in stations) + "]\n")
    return "".join(out), stations, view_angle_deg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    template_lines = open(TEMPLATE).readlines()
    for path in sorted(glob.glob("config/design/*.yaml")):
        name = os.path.splitext(os.path.basename(path))[0]
        text, stations, view_angle_deg = convert(path, template_lines)
        out_path = os.path.join(OUT_DIR, f"strawtubes_config_{name}.yaml")
        with open(out_path, "w") as f:
            f.write(text)
        z = ", ".join(f"{s:.3f}" for s in stations)
        print(f"{name}: view_angle = {view_angle_deg:.5g} deg,  stations (cm) = [{z}]  ->  {out_path}")


if __name__ == "__main__":
    main()
