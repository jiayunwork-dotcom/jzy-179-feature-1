"""输入拒收：逐项指出是哪个字段出的问题。"""
from __future__ import annotations

from conftest import bare_fuel, make_case


def _post(client, regions, **kw):
    body = dict(name="校验", regions=regions,
                left_boundary="zero_flux", right_boundary="zero_flux")
    body.update(kw)
    return client.post("/cases", json=body)


def _fields(r):
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    # FastAPI 字段级 422 用 detail 列表；物理聚合用 {error, details}
    if isinstance(detail, dict):
        return {d["field"] for d in detail["details"]}
    return {tuple(e["loc"]) for e in detail}


def test_thickness_must_be_positive(client):
    r = _post(client, [{**bare_fuel(50.0, 50), "thickness": 0.0}])
    fields = _fields(r)
    assert any("thickness" in str(f) for f in fields)


def test_D_must_be_positive(client):
    r = _post(client, [{**bare_fuel(50.0, 50), "D": -1.0}])
    fields = _fields(r)
    assert any("D" in str(f) for f in fields)


def test_sigma_a_nonnegative(client):
    r = _post(client, [{**bare_fuel(50.0, 50), "sigma_a": -0.01}])
    fields = _fields(r)
    assert any("sigma_a" in str(f) for f in fields)


def test_nu_sigma_f_nonnegative(client):
    r = _post(client, [{**bare_fuel(50.0, 50), "nu_sigma_f": -0.01}])
    fields = _fields(r)
    assert any("nu_sigma_f" in str(f) for f in fields)


def test_at_least_one_fissile_region(client):
    r = _post(client, [
        dict(thickness=20.0, D=1.0, sigma_a=0.1, nu_sigma_f=0.0, n_mesh=20),
        dict(thickness=10.0, D=1.0, sigma_a=0.05, nu_sigma_f=0.0, n_mesh=10),
    ])
    fields = _fields(r)
    assert any("regions" in str(f) for f in fields)
    msg = str(r.json()["detail"])
    assert "裂变" in msg


def test_too_many_regions(client):
    regions = [bare_fuel(1.0, 1) for _ in range(51)]
    regions[0]["nu_sigma_f"] = 0.12
    r = _post(client, regions)
    fields = _fields(r)
    assert any("regions" in str(f) for f in fields)
    assert "50" in str(r.json()["detail"])


def test_too_many_mesh_total(client):
    regions = [bare_fuel(50.0, 10_001),
               dict(thickness=10.0, D=1.0, sigma_a=0.1,
                    nu_sigma_f=0.12, n_mesh=10_000)]
    r = _post(client, regions)
    fields = _fields(r)
    assert any("regions" in str(f) for f in fields)
    assert "20000" in str(r.json()["detail"]) or "20_000" in str(r.json()["detail"])


def test_multiple_errors_listed_together(client):
    """一次给多处错误，要逐项都报出来，而不是只撞回第一个。"""
    r = _post(client, [
        dict(thickness=-1.0, D=0.0, sigma_a=-0.2, nu_sigma_f=-0.3, n_mesh=10),
    ])
    text = r.text
    for key in ("thickness", "D", "sigma_a", "nu_sigma_f"):
        assert key in text, f"错误清单缺少字段 {key}: {text}"


def test_invalid_region_rejected_on_patch_too(client):
    case = make_case(client, [bare_fuel(50.0, 50)])
    # 改完会导致无裂变材料 → 拒绝且不产生新版本
    r = client.patch(f"/cases/{case['id']}", json={
        "region_index": 0, "region": {"nu_sigma_f": 0.0}})
    assert r.status_code == 422
    got = client.get(f"/cases/{case['id']}").json()
    assert got["current_version"] == 1
    assert got["regions"][0]["nu_sigma_f"] == 0.12


def test_boundary_enum_validation(client):
    r = client.post("/cases", json=dict(
        name="x", regions=[bare_fuel(50.0, 50)],
        left_boundary="marshak", right_boundary="zero_flux"))
    assert r.status_code == 422
    assert "left_boundary" in r.text
