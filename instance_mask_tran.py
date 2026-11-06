import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from pycocotools import mask as maskUtils


# ==========================
# 路径
# ==========================

SOURCE_DIR = Path(
    "/data/liudenghui/dataset/stomach_intance1"
)


OUTPUT_DIR = Path(
    "/data/liudenghui/dataset/stomach_cellpose"
)


NUM_FOLDS = 5



# ==========================
# COCO RLE -> Cellpose instance mask
# ==========================

def convert_split(
        json_file,
        image_dir,
        output_image_dir,
        output_mask_dir
):

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(json_file, "r") as f:

        coco = json.load(f)



    # image_id -> annotations

    img_to_anns = {}

    for ann in coco["annotations"]:

        img_id = ann["image_id"]

        if img_id not in img_to_anns:

            img_to_anns[img_id] = []

        img_to_anns[img_id].append(ann)



    count = 0



    for img_info in coco["images"]:


        img_id = img_info["id"]

        file_name = Path(
            img_info["file_name"]
        ).name


        img_path = (
            image_dir /
            file_name
        )


        if not img_path.exists():

            print(
                "Missing image:",
                img_path
            )

            continue



        # 读取尺寸

        img = cv2.imread(
            str(img_path)
        )


        h, w = img.shape[:2]



        # ==========================
        # instance mask
        # ==========================

        instance_mask = np.zeros(
            (h, w),
            dtype=np.uint16
        )


        anns = img_to_anns.get(
            img_id,
            []
        )


        instance_id = 1

        for ann in anns:

            segmentation = ann["segmentation"]

            decoded_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            # =====================
            # polygon
            # =====================

            if isinstance(segmentation, list):

                for poly in segmentation:
                    pts = np.array(
                        poly,
                        dtype=np.float32
                    )

                    pts = pts.reshape(
                        -1,
                        2
                    )

                    pts = np.round(
                        pts
                    ).astype(
                        np.int32
                    )

                    cv2.fillPoly(
                        decoded_mask,
                        [pts],
                        1
                    )


            # =====================
            # RLE
            # =====================

            elif isinstance(segmentation, dict):

                decoded_mask = maskUtils.decode(
                    segmentation
                )

                if decoded_mask.ndim == 3:
                    decoded_mask = decoded_mask[:, :, 0]


            else:

                print(
                    "未知格式:",
                    type(segmentation)
                )

                continue

            instance_mask[
                decoded_mask > 0
                ] = instance_id

            instance_id += 1



        # ==========================
        # 保存图片
        # ==========================

        shutil.copy2(
            img_path,
            output_image_dir / file_name
        )


        # ==========================
        # 保存instance mask
        # ==========================

        mask_name = (
            Path(file_name).stem
            +
            "_masks.png"
        )


        cv2.imwrite(
            str(
                output_mask_dir /
                mask_name
            ),
            instance_mask
        )


        print(
            file_name,
            "instances:",
            instance_id - 1,
            "max label:",
            instance_mask.max()
        )


        count += 1



    print(
        "Finished:",
        json_file,
        "images:",
        count
    )




# ==========================
# 五折转换
# ==========================

def main():


    for fold in range(NUM_FOLDS):


        print(
            "\n========== Fold",
            fold,
            "=========="
        )


        fold_src = (
            SOURCE_DIR /
            f"fold_{fold}"
        )


        fold_dst = (
            OUTPUT_DIR /
            f"fold_{fold}"
        )


        for split in [
            "train",
            "val"
        ]:


            json_file = (
                fold_src
                /
                split
                /
                "annotations"
                /
                f"instances_{split}.json"
            )


            image_dir = (
                fold_src
                /
                split
                /
                "images"
            )


            out_image_dir = (
                fold_dst
                /
                split
                /
                "images"
            )


            out_mask_dir = (
                fold_dst
                /
                split
                /
                "masks"
            )


            convert_split(
                json_file,
                image_dir,
                out_image_dir,
                out_mask_dir
            )



if __name__ == "__main__":

    main()