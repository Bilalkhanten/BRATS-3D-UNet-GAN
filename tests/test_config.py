import json

from bratsgan.config import DEFAULTS, load_config, parse_overrides


def test_overrides_parse_json_and_strings():
    o = parse_overrides(["epochs_seg=5", "patch=[16,16,16]", "name=demo", "amp=false", "lr=1e-3"])
    assert o == {"epochs_seg": 5, "patch": [16, 16, 16], "name": "demo", "amp": False, "lr": 1e-3}


def test_load_config_precedence(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"base": 8, "lr": 0.1}))
    cfg = load_config(str(f), {"lr": 0.2})
    assert cfg["base"] == 8 and cfg["lr"] == 0.2 and cfg["depth"] == DEFAULTS["depth"]


def test_all_shipped_configs_use_known_keys():
    from pathlib import Path

    for p in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")):
        unknown = set(json.loads(p.read_text())) - set(DEFAULTS)
        assert not unknown, f"{p.name}: unknown keys {unknown}"


def test_ablation_chain_is_cumulative():
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "configs"
    c = {n: load_config(str(d / f"{n}.json")) for n in ("a0_unet", "a1_res", "a2_att", "a3_maf", "a4_tversky", "a5_full")}
    assert not c["a0_unet"]["residual"] and c["a1_res"]["residual"]
    assert not c["a1_res"]["attention"] and c["a2_att"]["attention"]
    assert c["a2_att"]["fusion"] == "early" and c["a3_maf"]["fusion"] == "maf"
    assert c["a3_maf"]["w_tv"] == 0 and c["a4_tversky"]["w_tv"] > 0
    assert not c["a4_tversky"]["gan"] and c["a5_full"]["gan"]
