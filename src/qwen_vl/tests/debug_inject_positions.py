"""
Debug: _inject_query_tokens  (Logic B = 3DRS-FORK, ONE consolidated block)
==========================================================================
Standalone replica (only needs torch) of the NEW one-block injection in
modeling_qwen2_5_vl.py. Run on any machine to verify the injected layout:

    [pre-text] [blank?] [ALL image tokens] [geometry?] [depth?] [post-text]

Each query block has n_frames*query_size rows (frame-major), sliced from a
(max_frames*query_size, H) parameter. This is faithful to the fork's
llava_qwen.py (32*query_size appended once), NOT the old per-frame scatter.

Usage:
    python src/qwen_vl/tests/debug_inject_positions.py
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ── Standalone replica of the NEW _inject_query_tokens ─────────────────────────
def inject_query_tokens(
    inputs_embeds: torch.Tensor,    # (B,T,H)
    attention_mask: torch.Tensor,   # (B,T)
    position_ids: torch.Tensor,     # (3,B,T)
    labels: Optional[torch.Tensor], # (B,T) or None
    visual_mask: torch.Tensor,      # (B,T) bool
    query_type: str,
    query_size: int,
    blank_query: Optional[nn.Parameter],
    geometry_query: Optional[nn.Parameter],
    depth_query: Optional[nn.Parameter],
) -> Tuple:
    """Exact copy of the model's one-block _inject_query_tokens logic."""
    IGNORE_INDEX = -100
    qt = query_type
    qs = query_size
    dev, dtype = inputs_embeds.device, inputs_embeds.dtype
    B, T, H = inputs_embeds.shape

    e_out, a_out, p_out, l_out, gm_out, dm_out = [], [], [], [], [], []

    for b in range(B):
        se = inputs_embeds[b]; sa = attention_mask[b]; sp = position_ids[:, b, :]
        sl = labels[b] if labels is not None else None
        sv = visual_mask[b]

        # contiguous visual segments (= frames in Qwen multi-image)
        segs, in_blk = [], False
        for t in range(T):
            if sv[t] and not in_blk:
                in_blk, blk_s = True, t
            elif not sv[t] and in_blk:
                in_blk = False; segs.append((blk_s, t))
        if in_blk:
            segs.append((blk_s, T))

        if not segs:
            e_out.append(se); a_out.append(sa); p_out.append(sp)
            if sl is not None: l_out.append(sl)
            gm_out.append(torch.zeros(T, dtype=torch.bool, device=dev))
            dm_out.append(torch.zeros(T, dtype=torch.bool, device=dev))
            continue

        el, al, ptl, phl, pwl, ll, gml, dml = [], [], [], [], [], [], [], []
        n_frames = len(segs)
        first_start = segs[0][0]
        last_end = segs[-1][1]
        n_req = n_frames * qs

        def _emit_raw(a, c):
            if c <= a:
                return
            n = c - a
            el.append(se[a:c]); al.append(sa[a:c])
            ptl.append(sp[0, a:c]); phl.append(sp[1, a:c]); pwl.append(sp[2, a:c])
            if sl is not None: ll.append(sl[a:c])
            gml.append(torch.zeros(n, dtype=torch.bool, device=dev))
            dml.append(torch.zeros(n, dtype=torch.bool, device=dev))

        def _emit_query(param, pt, ph, pw, is_geo, is_depth):
            block = param[:n_req].to(dev, dtype)
            n = block.shape[0]
            el.append(block); al.append(torch.ones(n, device=dev, dtype=sa.dtype))
            ptl.append(torch.full((n,), pt, device=dev, dtype=sp.dtype))
            phl.append(torch.full((n,), ph, device=dev, dtype=sp.dtype))
            pwl.append(torch.full((n,), pw, device=dev, dtype=sp.dtype))
            if sl is not None: ll.append(torch.full((n,), IGNORE_INDEX, device=dev, dtype=sl.dtype))
            gml.append(torch.full((n,), is_geo, dtype=torch.bool, device=dev))
            dml.append(torch.full((n,), is_depth, dtype=torch.bool, device=dev))

        vt = sp[0, last_end - 1].item(); vh = sp[1, last_end - 1].item(); vw = sp[2, last_end - 1].item()

        _emit_raw(0, first_start)
        if qt and 'blank' in qt and blank_query is not None:
            bt = sp[0, first_start].item(); bh = sp[1, first_start].item(); bw = sp[2, first_start].item()
            _emit_query(blank_query, bt, bh, bw, False, False)
        _emit_raw(first_start, last_end)
        if qt and 'geometry' in qt and geometry_query is not None:
            _emit_query(geometry_query, vt, vh, vw, True, False)
        if qt and 'depth' in qt and depth_query is not None:
            _emit_query(depth_query, vt, vh, vw, False, True)
        _emit_raw(last_end, T)

        e_out.append(torch.cat(el, 0)); a_out.append(torch.cat(al, 0))
        p_out.append(torch.stack([torch.cat(ptl), torch.cat(phl), torch.cat(pwl)], 0))
        if sl is not None: l_out.append(torch.cat(ll, 0))
        gm_out.append(torch.cat(gml, 0)); dm_out.append(torch.cat(dml, 0))

    new_e = torch.stack(e_out, 0)
    new_a = torch.stack(a_out, 0)
    new_p = torch.stack(p_out, 0).permute(1, 0, 2)
    new_l = torch.stack(l_out, 0) if l_out else None
    new_gm = torch.stack(gm_out, 0)
    new_dm = torch.stack(dm_out, 0)
    return new_e, new_a, new_p, new_l, new_gm, new_dm


def _classify(new_e, gm, dm, vis_mask, T_orig):
    """Label each output token by type using the traceable id in embed[...,0]."""
    types = []
    T_new = new_e.shape[1]
    for i in range(T_new):
        if gm[0, i]:
            types.append('GEO'); continue
        if dm[0, i]:
            types.append('DEP'); continue
        val = new_e[0, i, 0].item()
        if val % 1000 == 0 and 0 <= int(val / 1000) < T_orig:
            o = int(val / 1000)
            types.append('VIS' if vis_mask[0, o] else 'TXT')
        else:
            types.append('BLK')
    return types


def _make(T, vis_ranges, MF=32, QS=2, H=8):
    B = 1
    embeds = torch.randn(B, T, H)
    for i in range(T):
        embeds[0, i, 0] = i * 1000          # traceable id
    attn = torch.ones(B, T, dtype=torch.long)
    pos = torch.arange(T).view(1, 1, T).expand(3, B, T).clone()
    labels = torch.arange(100, 100 + T).view(1, T)
    vis = torch.zeros(B, T, dtype=torch.bool)
    for (a, c) in vis_ranges:
        vis[0, a:c] = True
    blk = nn.Parameter(torch.randn(MF * QS, H) * 0.01)
    geo = nn.Parameter(torch.randn(MF * QS, H) * 0.01)
    dep = nn.Parameter(torch.randn(MF * QS, H) * 0.01)
    return embeds, attn, pos, labels, vis, blk, geo, dep


def test_two_frames_one_block():
    """KEY test: 2 frames must produce ONE geometry block of 2*QS AFTER the last
    frame (not scattered between frames)."""
    print("\n" + "#" * 70)
    print("  Two frames -> ONE consolidated geometry+depth block (QS=2)")
    print("  Layout in:  [T T][V*4][T T][V*3][T T]")
    print("  Expect out: [T T][V*4][T T][V*3][GEO*4][DEP*4][T T]")
    print("#" * 70)
    QS = 2; T = 13
    embeds, attn, pos, labels, vis, blk, geo, dep = _make(T, [(2, 6), (8, 11)], QS=QS)
    ne, na, npos, nl, gm, dm = inject_query_tokens(
        embeds, attn, pos, labels, vis, "geometry,depth", QS, None, geo, dep)
    types = _classify(ne, gm, dm, vis, T)
    print("  out:", " ".join(types))
    n_frames = 2
    ok = True
    # exactly one contiguous GEO run, of length n_frames*QS, right after last VIS
    geo_idx = [i for i, t in enumerate(types) if t == 'GEO']
    dep_idx = [i for i, t in enumerate(types) if t == 'DEP']
    if len(geo_idx) != n_frames * QS: print(f"  [FAIL] GEO count {len(geo_idx)} != {n_frames*QS}"); ok = False
    if len(dep_idx) != n_frames * QS: print(f"  [FAIL] DEP count {len(dep_idx)} != {n_frames*QS}"); ok = False
    if geo_idx and geo_idx != list(range(geo_idx[0], geo_idx[0] + len(geo_idx))): print("  [FAIL] GEO not contiguous"); ok = False
    if dep_idx and dep_idx != list(range(dep_idx[0], dep_idx[0] + len(dep_idx))): print("  [FAIL] DEP not contiguous"); ok = False
    # GEO must come immediately after the LAST visual token, DEP after GEO
    last_vis = max(i for i, t in enumerate(types) if t == 'VIS')
    if geo_idx and geo_idx[0] != last_vis + 1: print("  [FAIL] GEO not right after last VIS"); ok = False
    if geo_idx and dep_idx and dep_idx[0] != geo_idx[-1] + 1: print("  [FAIL] DEP not right after GEO"); ok = False
    # original token order preserved
    orig = [int(ne[0, i, 0].item() / 1000) for i, t in enumerate(types) if t in ('VIS', 'TXT')]
    if orig != sorted(orig): print("  [FAIL] original order broken"); ok = False
    print("  Result:", "PASS" if ok else "FAIL")
    return ok


def test_blank_geo_depth_order():
    """blank prepended before ALL images; geo+depth appended after."""
    print("\n" + "#" * 70)
    print("  blank+geometry+depth, single frame (QS=2)")
    print("  Expect: [T*3][BLK*2][V*4][GEO*2][DEP*2][T*3]")
    print("#" * 70)
    QS = 2; T = 10
    embeds, attn, pos, labels, vis, blk, geo, dep = _make(T, [(3, 7)], QS=QS)
    ne, na, npos, nl, gm, dm = inject_query_tokens(
        embeds, attn, pos, labels, vis, "blank,geometry,depth", QS, blk, geo, dep)
    types = _classify(ne, gm, dm, vis, T)
    expect = ['TXT']*3 + ['BLK']*2 + ['VIS']*4 + ['GEO']*2 + ['DEP']*2 + ['TXT']*3
    print("  out:   ", " ".join(types))
    print("  expect:", " ".join(expect))
    ok = (types == expect)
    # labels at query positions must be IGNORE_INDEX (-100)
    q_pos = [i for i, t in enumerate(types) if t in ('BLK', 'GEO', 'DEP')]
    if any(nl[0, i].item() != -100 for i in q_pos): print("  [FAIL] query labels not IGNORE_INDEX"); ok = False
    print("  Result:", "PASS" if ok else "FAIL")
    return ok


def test_count_scales_with_frames():
    """4 frames, QS=4 -> 16 geo tokens (matches 3DRS 32*qs at 32 frames -> 128)."""
    print("\n" + "#" * 70)
    print("  Count check: 4 frames x QS=4 -> 16 geometry tokens")
    print("#" * 70)
    QS = 4; T = 24
    ranges = [(1, 5), (7, 11), (13, 17), (19, 23)]
    embeds, attn, pos, labels, vis, blk, geo, dep = _make(T, ranges, QS=QS)
    ne, na, npos, nl, gm, dm = inject_query_tokens(
        embeds, attn, pos, labels, vis, "geometry", QS, None, geo, dep)
    ok = (gm.sum().item() == 4 * QS) and (dm.sum().item() == 0) and (ne.shape[1] == T + 4 * QS)
    print(f"  geo={gm.sum().item()} (expect {4*QS}), added={ne.shape[1]-T} (expect {4*QS})")
    print("  Result:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("  _inject_query_tokens  (NEW one-block / 3DRS-fork) verification")
    print("=" * 70)
    tests = [
        ("two frames -> one block", test_two_frames_one_block),
        ("blank+geo+depth order", test_blank_geo_depth_order),
        ("count scales with frames", test_count_scales_with_frames),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback; traceback.print_exc(); results[name] = False
    print("\n" + "=" * 70 + "\n  SUMMARY")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"\n  {n_ok}/{len(results)} passed")
    if n_ok < len(results):
        import sys; sys.exit(1)
