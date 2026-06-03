"""Unit tests for io / laterality / association helpers (no real data needed)."""
import numpy as np
import pandas as pd
import nibabel as nib

from etfvbm.io import _norm_subject
from etfvbm.laterality import flip_to_treated, align_hemispheres
from etfvbm.association import fdr_bh


def test_norm_subject():
    assert _norm_subject("sub-ET001") == "ET001"
    assert _norm_subject("ET001") == "ET001"


def test_flip_to_treated():
    arr = np.zeros((4, 3, 2), np.float32)
    arr[0] = 1.0  # signal on the x=0 (left) edge
    img = nib.Nifti1Image(arr, np.eye(4))
    # treated on R, target L -> flips, so signal moves to x=-1 (right edge)
    flipped = np.asarray(flip_to_treated(img, "R", target_side="L").dataobj)
    assert flipped[-1].sum() == arr[0].sum()
    # treated already L -> no-op
    same = np.asarray(flip_to_treated(img, "L", target_side="L").dataobj)
    assert np.array_equal(same, arr)


def test_align_hemispheres_ipsi_contra():
    # one subject treated L, one treated R; Left lobule bigger than Right
    cereb = pd.DataFrame({
        "subject": ["A", "B"], "session": ["ses-preop", "ses-preop"],
        "GM_Left_V": [10.0, 10.0], "GM_Right_V": [6.0, 6.0],
    })
    clin = pd.DataFrame({"subject": ["A", "B"], "treated_side": ["L", "R"]})
    out = align_hemispheres(cereb, clin)
    # subject A treated L -> ipsi = Left(10); subject B treated R -> ipsi = Right(6)
    assert out.loc[out.subject == "A", "IPSI_V"].iloc[0] == 10.0
    assert out.loc[out.subject == "B", "IPSI_V"].iloc[0] == 6.0
    assert out.loc[out.subject == "A", "CONTRA_V"].iloc[0] == 6.0


def test_align_hemispheres_unknown_side_is_nan():
    cereb = pd.DataFrame({"subject": ["A"], "session": ["ses-preop"],
                          "GM_Left_V": [10.0], "GM_Right_V": [6.0]})
    clin = pd.DataFrame({"subject": ["A"], "treated_side": [None]})
    out = align_hemispheres(cereb, clin)
    assert np.isnan(out["IPSI_V"].iloc[0])
    # side-agnostic fallback still present
    assert out["Left_hemi_GM"].iloc[0] == 10.0


def test_fdr_bh_monotone():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    q = fdr_bh(p)
    assert np.all(np.isfinite(q))
    assert np.all(q >= p)  # adjusted >= raw
