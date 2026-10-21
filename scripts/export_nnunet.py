"""Export the SAME patient-level splits to nnU-Net v2 so that the strongest baseline is evaluated
on identical data.  Requires `pip install nnunetv2` and the raw BraTS folder.

    python scripts/export_nnunet.py --root /data/BraTS2020 --splits splits.json --nnunet-raw $nnUNet_raw --dataset-id 501

Afterwards run (per fold k of 0..4):
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    nnUNetv2_train 501 3d_fullres k
    nnUNetv2_predict -i <imagesTs> -o <out> -d 501 -c 3d_fullres -f k
The generated splits_final.json forces nnU-Net to use our folds; the 'test' cases are exported as imagesTs.
"""
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bratsgan.data import find_cases
from bratsgan.utils import read_json

MOD = ["t1", "t1ce", "t2", "flair"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--nnunet-raw", required=True); ap.add_argument("--dataset-id", type=int, default=501)
    a = ap.parse_args()
    splits = read_json(a.splits)
    cases = {c["id"]: c for c in find_cases(a.root)}
    ds = Path(a.nnunet_raw) / f"Dataset{a.dataset_id:03d}_BraTS2020"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    pool = [i for f in splits["folds"] for i in f]
    import nibabel as nib, numpy as np

    def put(cid, img_dir, lab_dir):
        c = cases[cid]
        for j, m in enumerate(MOD):
            shutil.copy(c["paths"][m], ds / img_dir / f"{cid}_{j:04d}.nii.gz")
        seg = nib.load(c["seg"]); arr = np.asarray(seg.dataobj).astype(np.uint8); arr[arr == 4] = 3   # nnU-Net wants contiguous labels
        nib.save(nib.Nifti1Image(arr, seg.affine), str(ds / lab_dir / f"{cid}.nii.gz"))

    for cid in pool:
        put(cid, "imagesTr", "labelsTr")
    for cid in splits["test"]:
        put(cid, "imagesTs", "labelsTs")
    # Region-based training. After remapping 4->3 the labels are 1=NCR, 2=ED, 3=ET, so
    # WT = {1,2,3}, TC = {1,3}, ET = {3}.  (Check the region syntax against the nnU-Net docs of your installed version.)
    dj = {"channel_names": {str(j): m for j, m in enumerate(MOD)},
          "labels": {"background": 0, "whole_tumor": [1, 2, 3], "tumor_core": [1, 3], "enhancing_tumor": [3]},
          "regions_class_order": [1, 2, 3], "numTraining": len(pool), "file_ending": ".nii.gz"}
    (ds / "dataset.json").write_text(json.dumps(dj, indent=2))
    nn_splits = [{"train": [i for j, f in enumerate(splits["folds"]) if j != k for i in f], "val": splits["folds"][k]} for k in range(len(splits["folds"]))]
    pre = Path(a.nnunet_raw).parent / "nnUNet_preprocessed" / ds.name
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(json.dumps(nn_splits, indent=2))
    print(f"exported to {ds}; splits_final.json written to {pre}")


if __name__ == "__main__":
    main()
