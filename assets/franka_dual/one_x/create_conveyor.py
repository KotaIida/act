import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_capsule_belt(num_capsules, capsule_length=0.2, capsule_radius=0.02, spacing=0.025):
    # ルート要素<mujoco>
    mujoco = ET.Element('mujoco', model='conveyor_belt')

    # <worldbody>を作成
    worldbody = ET.SubElement(mujoco, 'worldbody')
    compiler = ET.SubElement(mujoco, "compiler", {"meshdir": "assets", "texturedir":"assets"})
    asset = ET.SubElement(mujoco, "asset")
    texture = ET.SubElement(asset, "texture", {"type": "2d", "file":"iron.png"})
    material = ET.SubElement(asset, "material", {"name": "onex_poll", "texture": "iron", "reflectance": "0.5"})


    body = ET.SubElement(worldbody, 'body', {"name": "side", "pos":"-1.4875 0 0.678"})
    back_side = ET.SubElement(body, 'geom', {"type": "box", "pos": "1.475 0.2 0", "size":"1.5 0.015 0.022", "rgba": "0.00784314 0.52941176 0.65882353 1"})
    front_side = ET.SubElement(body, 'geom', {"type": "box", "pos": "1.475 -0.2 0", "size":"1.5 0.015 0.022",  "rgba": "0.00784314 0.52941176 0.65882353 1"})
    left_front_leg = ET.SubElement(body, 'geom', {"type": "box", "pos": "0 -0.2 -0.35", "size":"0.022 0.022 0.35", "material": "onex_poll"})
    left_back_leg = ET.SubElement(body, 'geom', {"type": "box", "pos": "0 0.2 -0.35", "size":"0.022 0.022 0.35", "material": "onex_poll"})
    right_front_leg = ET.SubElement(body, 'geom', {"type": "box", "pos": "2.95 -0.2 -0.35", "size":"0.022 0.022 0.35", "material": "onex_poll"})
    right_back_leg = ET.SubElement(body, 'geom', {"type": "box", "pos": "2.95 0.2 -0.35", "size":"0.022 0.022 0.35", "material": "onex_poll"})


    # body要素を作成
    body = ET.SubElement(worldbody, 'body', {'pos': f'-1.4875 0 0.678'})
    # 円柱をnum_capsules個作成
    for i in range(num_capsules):
        # 各円柱の位置を計算
        x_pos = i * spacing*2
        
        # geom要素（capsule）を作成
        geom = ET.SubElement(body, 'geom', {
            'type': 'cylinder',
            'pos': f'{x_pos} 0 0 ',
            'size': f"{str(capsule_radius)} {capsule_length}",
            "euler": f"90 0 0",
            "material": "onex_poll"
        })
        
    # XMLをきれいにフォーマットして出力
    rough_string = ET.tostring(mujoco, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # XMLファイルに書き出し
    with open("conveyor_belt.xml", "w") as f:
        f.write(pretty_xml)

# 例として10個の円柱を並べる
create_capsule_belt(num_capsules=60)