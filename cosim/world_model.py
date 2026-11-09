"""Programmatic MJCF construction for the factory-cell world simulator."""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARM_MODEL_PATH = ROOT / "sim" / "assets" / "arm_table.xml"


def _values(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _append_material(asset: ET.Element, name: str, rgba: str) -> None:
    ET.SubElement(asset, "material", name=name, rgba=rgba)


def _factory_assets(source: ET.Element) -> ET.Element:
    asset = copy.deepcopy(source.find("asset"))
    if asset is None:
        asset = ET.Element("asset")
    _append_material(asset, "world_floor", "0.30 0.32 0.34 1")
    _append_material(asset, "world_wall", "0.68 0.71 0.75 1")
    _append_material(asset, "world_rack", "0.20 0.34 0.48 1")
    _append_material(asset, "world_fence", "0.85 0.58 0.12 1")
    _append_material(asset, "world_table", "0.42 0.31 0.22 1")
    _append_material(asset, "world_amr", "0.12 0.48 0.68 1")
    _append_material(asset, "world_deck", "0.10 0.13 0.16 1")
    _append_material(asset, "world_crate", "0.76 0.38 0.10 1")
    _append_material(asset, "world_load", "0.12 0.72 0.32 0.65")
    _append_material(asset, "world_drop", "0.72 0.18 0.18 0.65")
    return asset


def _obstacle_material(name: str) -> str:
    if name == "drop_table":
        return "world_table"
    if "fence" in name:
        return "world_fence"
    if "pedestal" in name:
        return "joint"
    return "world_rack"


def _add_static_box(worldbody: ET.Element, item: dict[str, Any], material: str) -> None:
    sx, sy = item["size"]
    height = item["h"]
    x, y = item["center"]
    ET.SubElement(
        worldbody,
        "geom",
        name=item["name"],
        type="box",
        pos=_values((x, y, height / 2)),
        size=_values((sx / 2, sy / 2, height / 2)),
        material=material,
        friction="0.9 0.01 0.001",
    )


def _add_station_markers(worldbody: ET.Element, layout: dict[str, Any]) -> None:
    for station, material in (("LOAD", "world_load"), ("DROP", "world_drop")):
        x, y, _ = layout["stations"][station]
        ET.SubElement(
            worldbody,
            "geom",
            name=f"station_{station}",
            type="cylinder",
            pos=_values((x, y, 0.006)),
            size="0.32 0.006",
            material=material,
            contype="0",
            conaffinity="0",
            group="4",
        )


def _add_cameras(worldbody: ET.Element, layout: dict[str, Any]) -> None:
    floor_x, floor_y = layout["floor"]["size"]
    overview_target = ET.SubElement(
        worldbody,
        "body",
        name="overview_target",
        pos=_values((floor_x / 2, floor_y / 2, 0.25)),
    )
    ET.SubElement(overview_target, "site", name="overview_target_site", size="0.001", rgba="0 0 0 0")
    ET.SubElement(
        worldbody,
        "camera",
        name="overview",
        pos=_values((floor_x / 2, -2.6, 8.2)),
        mode="targetbody",
        target="overview_target",
        fovy="45",
    )

    arm_x, arm_y, arm_z = layout["arm"]["base"]
    load_x, load_y, _ = layout["stations"]["LOAD"]
    arm_target = ET.SubElement(
        worldbody,
        "body",
        name="arm_camera_target",
        pos=_values(((arm_x + load_x) / 2, (arm_y + load_y) / 2, arm_z + 0.3)),
    )
    ET.SubElement(arm_target, "site", name="arm_camera_target_site", size="0.001", rgba="0 0 0 0")
    ET.SubElement(
        worldbody,
        "camera",
        name="arm_cam",
        pos=_values((arm_x + 2.2, arm_y - 2.1, arm_z + 2.0)),
        mode="targetbody",
        target="arm_camera_target",
        fovy="48",
    )


def _add_amr(worldbody: ET.Element, layout: dict[str, Any]) -> None:
    footprint_x, footprint_y = layout["amr"]["footprint"]
    deck_height = float(layout["amr"]["deck_height"])
    amr = ET.SubElement(worldbody, "body", name="amr")
    ET.SubElement(amr, "joint", name="amr_x", type="slide", axis="1 0 0", limited="false")
    ET.SubElement(amr, "joint", name="amr_y", type="slide", axis="0 1 0", limited="false")
    ET.SubElement(amr, "joint", name="amr_yaw", type="hinge", axis="0 0 1", limited="false")

    chassis_half_z = max(0.04, (deck_height - 0.06) / 2)
    chassis_center_z = deck_height - 0.04 - chassis_half_z
    ET.SubElement(
        amr,
        "geom",
        name="amr_chassis",
        type="box",
        pos=_values((0, 0, chassis_center_z)),
        size=_values((footprint_x * 0.46, footprint_y * 0.46, chassis_half_z)),
        material="world_amr",
        mass="35",
        friction="1 0.01 0.001",
    )
    ET.SubElement(
        amr,
        "geom",
        name="amr_deck",
        type="box",
        pos=_values((0, 0, deck_height - 0.02)),
        size=_values((footprint_x / 2, footprint_y / 2, 0.02)),
        material="world_deck",
        mass="5",
        friction="1 0.01 0.001",
    )
    ET.SubElement(
        amr,
        "site",
        name="amr_lidar",
        pos=_values((0, 0, layout["lidar"]["height"])),
        size="0.025 0.012",
        type="cylinder",
        rgba="0.08 0.08 0.08 1",
    )


def _add_crate(worldbody: ET.Element, layout: dict[str, Any]) -> None:
    load_x, load_y, _ = layout["stations"]["LOAD"]
    sx, sy, sz = layout["crate"]["size"]
    transfer_z = float(layout["amr"]["deck_height"]) + sz / 2
    crate = ET.SubElement(
        worldbody,
        "body",
        name="crate",
        mocap="true",
        pos=_values((load_x, load_y, transfer_z)),
    )
    ET.SubElement(
        crate,
        "geom",
        name="crate_geom",
        type="box",
        size=_values((sx / 2, sy / 2, sz / 2)),
        material="world_crate",
        contype="0",
        conaffinity="0",
    )


def build_world_xml(
    layout: dict[str, Any], arm_model_path: str | Path = ARM_MODEL_PATH
) -> str:
    """Return a complete MJCF model assembled from the shared layout and arm model."""
    source = ET.parse(arm_model_path).getroot()
    root = ET.Element("mujoco", model="factory_cell_world")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
    )
    visual = source.find("visual")
    if visual is not None:
        root.append(copy.deepcopy(visual))
    root.append(_factory_assets(source))
    defaults = source.find("default")
    if defaults is not None:
        root.append(copy.deepcopy(defaults))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        name="world_key",
        pos="4 -2 10",
        dir="0.15 0.35 -1",
        diffuse="0.75 0.75 0.75",
        castshadow="true",
    )
    floor_x, floor_y = layout["floor"]["size"]
    ET.SubElement(
        worldbody,
        "geom",
        name="factory_floor",
        type="plane",
        pos=_values((floor_x / 2, floor_y / 2, -0.005)),
        size=_values((floor_x / 2 + 0.2, floor_y / 2 + 0.2, 0.05)),
        material="world_floor",
    )
    for wall in layout["walls"]:
        _add_static_box(worldbody, wall, "world_wall")
    for obstacle in layout["obstacles"]:
        _add_static_box(worldbody, obstacle, _obstacle_material(obstacle["name"]))
    _add_station_markers(worldbody, layout)
    _add_cameras(worldbody, layout)
    _add_amr(worldbody, layout)
    _add_crate(worldbody, layout)

    source_worldbody = source.find("worldbody")
    arm_base = source_worldbody.find("./body[@name='base']") if source_worldbody is not None else None
    if arm_base is None:
        raise ValueError(f"arm model {arm_model_path} has no body named 'base'")
    arm_base = copy.deepcopy(arm_base)
    arm_base.set("pos", _values(layout["arm"]["base"]))
    arm_base.set("euler", _values((0, 0, layout["arm"]["mount_yaw"])))
    worldbody.append(arm_base)

    for section_name in ("contact", "equality", "actuator"):
        section = source.find(section_name)
        if section is not None:
            root.append(copy.deepcopy(section))
    return ET.tostring(root, encoding="unicode")


__all__ = ["ARM_MODEL_PATH", "ROOT", "build_world_xml"]
