"""Tests for the Reentry Query Engine."""

import sys
import math
import numpy as np

sys.path.insert(0, ".")

from models.reentry_query_engine.archive_index import (
    ArchiveIndex,
    IdentityEntry,
    _hilbert_encode,
    _hilbert_bucket,
)
from models.reentry_query_engine.spatial_router import SpatialRouter
from models.reentry_query_engine.vector_searcher import VectorSearcher, VectorMatch
from models.reentry_query_engine.query_planner import QueryPlanner, QueryMode
from models.reentry_query_engine.commit_controller import CommitController, CommitDecision
from models.reentry_query_engine.metrics import ReentryMetrics
from models.reentry_query_engine.engine import ReentryQueryEngine


# -----------------------------------------------------------------------
# Hilbert curve
# -----------------------------------------------------------------------

def test_hilbert_basic():
    """Same coords → same key; different coords → different keys."""
    a = _hilbert_encode(100, 200, 16)
    b = _hilbert_encode(100, 200, 16)
    c = _hilbert_encode(500, 600, 16)
    assert a == b
    assert a != c


def test_hilbert_locality():
    """Nearby coordinates should produce nearby Hilbert keys (on average)."""
    close_diffs = []
    far_diffs = []
    for i in range(100):
        x, y = i * 10, i * 10
        key1 = _hilbert_encode(x, y, 16)
        key2 = _hilbert_encode(x + 5, y + 5, 16)
        close_diffs.append(abs(key1 - key2))
        key3 = _hilbert_encode(x + 500, y + 500, 16)
        far_diffs.append(abs(key1 - key3))
    assert np.mean(close_diffs) < np.mean(far_diffs)


# -----------------------------------------------------------------------
# ArchiveIndex
# -----------------------------------------------------------------------

def _make_entry(tid, x, y, frame, feat_dim=64):
    return IdentityEntry(
        track_id=tid,
        exit_x=x,
        exit_y=y,
        hilbert_key=0,
        exit_frame=frame,
        smooth_feat=np.random.randn(feat_dim).astype(np.float32),
    )


def test_archive_insert_delete():
    idx = ArchiveIndex(hilbert_order=8)
    e1 = _make_entry(1, 100, 200, 10)
    idx.insert(e1)
    assert idx.size == 1
    assert idx.get(1) is e1

    idx.delete(1)
    assert idx.size == 0
    assert idx.get(1) is None


def test_archive_expire():
    idx = ArchiveIndex(hilbert_order=8, max_gap=50)
    idx.insert(_make_entry(1, 100, 200, 10))
    idx.insert(_make_entry(2, 300, 400, 80))

    expired = idx.expire(current_frame=100)
    assert expired == 1  # entry 1 has gap=90 > 50
    assert idx.size == 1
    assert idx.get(2) is not None


def test_archive_spatial_candidates():
    idx = ArchiveIndex(hilbert_order=8)
    idx.insert(_make_entry(1, 100, 200, 10))
    idx.insert(_make_entry(2, 300, 400, 10))

    # Near entry 1
    cands = idx.spatial_candidates(105, 205, radius=1)
    ids = {e.track_id for e in cands}
    assert 1 in ids

    # Scan all
    all_entries = idx.scan_all()
    assert len(all_entries) == 2


def test_archive_duplicate_insert():
    idx = ArchiveIndex(hilbert_order=8)
    idx.insert(_make_entry(1, 100, 200, 10))
    idx.insert(_make_entry(1, 200, 300, 20))  # update same track_id
    assert idx.size == 1
    entry = idx.get(1)
    assert entry.exit_x == 200


# -----------------------------------------------------------------------
# SpatialRouter
# -----------------------------------------------------------------------

def test_spatial_router_expanding():
    idx = ArchiveIndex(hilbert_order=8)
    idx.insert(_make_entry(1, 100, 200, 10))
    router = SpatialRouter(base_radius=1, max_radius=4, min_candidates=1)
    cands = router.route(idx, 105.0, 205.0)
    assert len(cands) >= 1


def test_spatial_router_empty():
    idx = ArchiveIndex(hilbert_order=8)
    router = SpatialRouter()
    cands = router.route(idx, 100.0, 200.0)
    assert len(cands) == 0


# -----------------------------------------------------------------------
# VectorSearcher
# -----------------------------------------------------------------------

def test_vector_searcher_scoring():
    searcher = VectorSearcher(min_similarity=0.1)
    feat = np.random.randn(64).astype(np.float32)
    entry = _make_entry(1, 100, 200, 10, feat_dim=64)
    entry.smooth_feat = feat.copy()

    match = searcher.score_pair(entry, feat, 0.8, None, 12, 120)
    assert match is not None
    assert match.app_sim > 0.99  # same vector → ~1.0 cosine
    assert match.composite_score > 0.5


def test_vector_searcher_threshold():
    searcher = VectorSearcher(min_similarity=0.99, min_det_score=0.01)
    feat = np.random.randn(64).astype(np.float32)
    entry = _make_entry(1, 100, 200, 10, feat_dim=64)
    entry.smooth_feat = feat.copy()

    # Different vector → low cosine → below threshold
    match = searcher.score_pair(entry, np.random.randn(64).astype(np.float32), 0.8, None, 12, 120)
    assert match is None


def test_vector_searcher_search():
    searcher = VectorSearcher(min_similarity=0.1)
    feat = np.random.randn(64).astype(np.float32)
    entries = [_make_entry(i, 100 + i * 10, 200, 10, 64) for i in range(5)]
    for e in entries:
        e.smooth_feat = feat + np.random.randn(64).astype(np.float32) * 0.1

    matches = searcher.search(entries, feat, 0.8, None, 12, 120, top_k=3)
    assert len(matches) <= 3
    assert all(isinstance(m, VectorMatch) for m in matches)


# -----------------------------------------------------------------------
# QueryPlanner
# -----------------------------------------------------------------------

def test_planner_skip_empty():
    planner = QueryPlanner()
    plan = planner.plan(archive_size=0, det_score=0.8)
    assert plan.mode == QueryMode.SKIP


def test_planner_skip_weak_det():
    planner = QueryPlanner(min_det_score=0.5)
    plan = planner.plan(archive_size=100, det_score=0.3)
    assert plan.mode == QueryMode.SKIP


def test_planner_brute_force():
    planner = QueryPlanner(brute_force_threshold=50)
    plan = planner.plan(archive_size=30, det_score=0.8)
    assert plan.mode == QueryMode.BRUTE_FORCE


def test_planner_spatial_first():
    planner = QueryPlanner(brute_force_threshold=50)
    plan = planner.plan(archive_size=100, det_score=0.8)
    assert plan.mode == QueryMode.SPATIAL_FIRST
    assert plan.spatial_radius >= 1


def test_planner_gap_adaptive():
    planner = QueryPlanner()
    plan_small = planner.plan(archive_size=200, det_score=0.8, expected_gap=5)
    plan_large = planner.plan(archive_size=200, det_score=0.8, expected_gap=80)
    assert plan_large.spatial_radius >= plan_small.spatial_radius


# -----------------------------------------------------------------------
# CommitController
# -----------------------------------------------------------------------

def test_commit_pending():
    ctrl = CommitController(confirm_streak=2, confirm_gap=3, confirm_min_similarity=0.5)
    d = ctrl.propose(1, 0, 0.7, 10)
    assert d == CommitDecision.PENDING


def test_commit_confirmed():
    ctrl = CommitController(confirm_streak=2, confirm_gap=3, confirm_min_similarity=0.5)
    ctrl.propose(1, 0, 0.7, 10)
    d = ctrl.propose(1, 0, 0.7, 11)
    assert d == CommitDecision.CONFIRMED


def test_commit_reset_on_gap():
    ctrl = CommitController(confirm_streak=2, confirm_gap=2, confirm_min_similarity=0.5)
    ctrl.propose(1, 0, 0.7, 10)
    # Skip frame 11
    d = ctrl.propose(1, 0, 0.7, 12)
    assert d == CommitDecision.RESET


def test_commit_cleanup():
    ctrl = CommitController(confirm_streak=2, confirm_gap=2, confirm_min_similarity=0.5)
    ctrl.propose(1, 0, 0.7, 10)
    expired = ctrl.cleanup(current_frame=20, active_track_ids={1})
    assert len(expired) == 0 or ctrl.pending_count == 0


# -----------------------------------------------------------------------
# ReentryMetrics
# -----------------------------------------------------------------------

def test_metrics_summary():
    m = ReentryMetrics()
    m.begin_frame(10, 5)
    m.record_query(3, 1, 0.5)
    m.record_commit(5)
    m.record_recovery_attempt(5)
    m.end_frame()

    s = m.summary()
    assert s["total_queries"] == 1
    assert s["total_commits"] == 1
    assert s["frames"] == 1
    assert s["recovery_rate"] == 1.0


def test_metrics_gap_binned():
    m = ReentryMetrics()
    m.begin_frame(10, 1)
    m.record_recovery_attempt(3)
    m.record_commit(3)
    m.record_recovery_attempt(50)
    m.end_frame()

    s = m.summary()
    assert "1-5" in s["gap_binned"]
    assert s["gap_binned"]["1-5"]["commits"] == 1


# -----------------------------------------------------------------------
# ReentryQueryEngine (E2E)
# -----------------------------------------------------------------------

class MockTrack:
    def __init__(self, tid, x, y, w, h, frame, feat):
        self.track_id = tid
        self._tlwh = np.array([x, y, w, h], dtype=np.float32)
        self.frame_id = frame
        self.score = 0.9
        self.smooth_feat = feat
        self.tracklet_len = 50
        self.mean = np.array([x + w / 2, y + h / 2, w / (w + h), h, 1.0, -0.5, 0.01, -0.01], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 0.1


class MockDet:
    def __init__(self, x, y, w, h, score, feat):
        self._tlwh = np.array([x, y, w, h], dtype=np.float32)
        self.score = score
        self.curr_feat = feat


def test_engine_full_cycle():
    engine = ReentryQueryEngine(max_gap=120, hilbert_order=8, confirm_streak=2)
    feat = np.random.randn(128).astype(np.float32)

    # Archive a track
    track = MockTrack(1, 80, 180, 40, 40, 10, feat)
    engine.archive_track(track)
    assert engine.archive.size == 1

    # Query with a similar detection
    det_feat = feat + np.random.randn(128).astype(np.float32) * 0.05
    det = MockDet(85, 185, 40, 40, 0.85, det_feat)

    # Frame 12: first match → PENDING
    rec, pairs = engine.query([det], np.stack([det_feat]), 12)
    assert len(rec) == 0  # not confirmed yet

    # Frame 13: second match → CONFIRMED
    rec2, pairs2 = engine.query([det], np.stack([det_feat]), 13)
    assert len(rec2) == 1
    assert len(pairs2) == 1
    assert pairs2[0][0] == 1  # track_id
    assert engine.archive.size == 0  # removed from archive


def test_engine_no_match():
    engine = ReentryQueryEngine(max_gap=120)
    feat = np.random.randn(128).astype(np.float32)
    track = MockTrack(1, 80, 180, 40, 40, 10, feat)
    engine.archive_track(track)

    # Completely different feature
    det = MockDet(900, 900, 40, 40, 0.85, np.random.randn(128).astype(np.float32))
    rec, pairs = engine.query([det], np.stack([det.curr_feat]), 12)
    assert len(rec) == 0
    assert engine.archive.size == 1  # still in archive


def test_engine_expire():
    engine = ReentryQueryEngine(max_gap=50)
    feat = np.random.randn(128).astype(np.float32)
    track = MockTrack(1, 80, 180, 40, 40, 10, feat)
    engine.archive_track(track)

    n = engine.expire_entries(current_frame=100)
    assert n == 1
    assert engine.archive.size == 0


def test_engine_stats():
    engine = ReentryQueryEngine(max_gap=120)
    stats = engine.get_stats()
    assert stats["archive_size"] == 0
    assert "commit" in stats
    assert "metrics" in stats


def test_engine_handles_missing_features_without_misalignment():
    engine = ReentryQueryEngine(max_gap=120, confirm_streak=1)
    feat = np.random.randn(128).astype(np.float32)
    track = MockTrack(7, 50, 60, 30, 30, 5, feat)
    engine.archive_track(track)

    det_with_feat = MockDet(52, 62, 30, 30, 0.9, feat + np.random.randn(128).astype(np.float32) * 0.02)
    det_without_feat = MockDet(400, 400, 30, 30, 0.95, None)
    rec, pairs = engine.query(
        [det_with_feat, det_without_feat],
        [det_with_feat.curr_feat, None],
        8,
        det_ambiguities=[0.8, 0.1],
    )
    assert len(rec) == 0
    assert len(pairs) == 0

    rec2, pairs2 = engine.query(
        [det_with_feat, det_without_feat],
        [det_with_feat.curr_feat, None],
        9,
        det_ambiguities=[0.8, 0.1],
    )
    assert len(rec2) == 1
    assert pairs2[0][0] == 7


def test_engine_recent_rerank_prefers_more_recent_candidate_when_scores_close():
    engine = ReentryQueryEngine(
        recent_score_margin=0.03,
        recent_min_exit_frame_advantage=10,
    )
    old = _make_entry(1, 100, 100, 50)
    new = _make_entry(2, 100, 100, 70)
    matches = [
        VectorMatch(old, app_sim=0.90, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.820),
        VectorMatch(new, app_sim=0.89, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.810),
    ]
    best = engine._select_best_match(matches)
    assert best.entry.track_id == 2
    stats = engine.get_stats()["recent_rerank"]
    assert stats["considered"] == 1
    assert stats["swaps"] == 1


def test_engine_recent_rerank_keeps_top_when_recency_advantage_too_small():
    engine = ReentryQueryEngine(
        recent_score_margin=0.03,
        recent_min_exit_frame_advantage=30,
    )
    old = _make_entry(1, 100, 100, 50)
    new = _make_entry(2, 100, 100, 70)
    matches = [
        VectorMatch(old, app_sim=0.90, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.820),
        VectorMatch(new, app_sim=0.89, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.810),
    ]
    best = engine._select_best_match(matches)
    assert best.entry.track_id == 1


def test_engine_recent_rerank_keeps_top_when_score_gap_too_large():
    engine = ReentryQueryEngine(
        recent_score_margin=0.01,
        recent_min_exit_frame_advantage=10,
    )
    old = _make_entry(1, 100, 100, 50)
    new = _make_entry(2, 100, 100, 90)
    matches = [
        VectorMatch(old, app_sim=0.90, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.860),
        VectorMatch(new, app_sim=0.89, iou_sim=0.2, gap_factor=0.8, det_score=0.8, composite_score=0.830),
    ]
    best = engine._select_best_match(matches)
    assert best.entry.track_id == 1


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        sys.exit(1)
