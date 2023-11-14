from histolab.slide import Slide
from histolab.tiler import RandomTiler
from histolab.tiler import GridTiler
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer
import os

# BASE_PATH = os.getcwd()
CRC_path = './lusc/0/'  # 图片位置
slide_files = os.listdir(CRC_path)
path = "./segment/lusc/"  # 切割完成后图片存放的位置
i = 0
for slide in slide_files:
    i = i + 1
    slide_path = CRC_path + slide
    CRC_slide = Slide(slide_path, processed_path=path)
    print("(", i, "/", len(slide_files), ")", f"Slide name:{CRC_slide.name}")  # 幻灯片名称
    # print(f"Levels:{CRC_slide.levels}")
    # print(f"Dimensions at level 0:{CRC_slide.dimensions}")
    # print(f"Dimensions at level 1:{CRC_slide.level_dimensions(level=1)}")
    # print(f"Dimensions at level 2:{CRC_slide.level_dimensions(level=2)}")
    # print("Native magnification factor:", CRC_slide.level_magnification_factor())
    if CRC_slide.level_magnification_factor(level=0) == '20.0X' or CRC_slide.level_magnification_factor(level=0) == '40.0X':
        slide_level = 0
        print("Magnification factor corresponding to level 0:", CRC_slide.level_magnification_factor(level=slide_level))
    elif CRC_slide.level_magnification_factor(level=1) == '20.0X' or CRC_slide.level_magnification_factor(level=1) == '40.0X':
        slide_level = 1
        print("Magnification factor corresponding to level 1:", CRC_slide.level_magnification_factor(level=slide_level))
    else:
        print('Magnification factor has no 20.0X !')
        continue

    grid_tiles_extractor = GridTiler(
       tile_size=(256, 256),
       level=slide_level,
       check_tissue=False,
       pixel_overlap=0, # default
       prefix=slide[0:23]+'/', # save tiles in the "grid" subdirectory of slide's processed_path
       suffix=".png" # default
    )
    grid_tiles_extractor.extract(CRC_slide)  # 是否存切割后的图像

