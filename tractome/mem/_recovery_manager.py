import logging

import numpy as np

from tractome.compute import (
    FINE_Q,
    KEEP_FLOOR,
    RECOVERY_NPROBE,
    SUBSAMPLE_THRESHOLD,
    fine_cluster_labels,
    knn_indices,
    stratified_subsample,
)

FINE_LABELS_PREFIX = "fine_labels"


def fine_labels_key(embedding_name):
    """Return the ``data_per_streamline`` key caching strata for an embedding.

    Fine-stratum labels are produced by a MiniBatchKMeans pass over one
    specific embedding, so they are only meaningful alongside that embedding.
    Qualifying the key by embedding name keeps labels built from one embedding
    from being reused when another is selected.

    Parameters
    ----------
    embedding_name : str
        The ``data_per_streamline`` key of the embedding the labels describe.

    Returns
    -------
    str
        The cache key for that embedding's fine-stratum labels.
    """
    return f"{FINE_LABELS_PREFIX}__{embedding_name}"


class RecoveryManager:
    """Manage the subsampled working set and the fiber-recovery index.

    A large tractogram is reduced to a stratified working set at load time (see
    :func:`tractome.compute.stratified_subsample`). This singleton keeps the
    per-streamline fine-stratum ``labels`` produced by that one-time pass so the
    un-sampled streamlines can be recovered on demand: siblings of a selection
    within a fine stratum are its embedding-space neighbours.

    State held here is *global per tractogram* (not per undo/redo state):

    labels : ndarray or None
        ``int32`` stratum id of every streamline in the full tractogram.
    medoid_ids : ndarray or None
        Medoid streamline id of each non-empty stratum.
    available_mask : ndarray or None
        Boolean mask over the full set: ``True`` where a streamline exists but
        is neither in the base sample nor already recovered. Cleared as ids are
        handed out; intentionally monotonic within a session.
    active : bool
        ``True`` only when subsampling actually happened (``N > threshold``).
    embedding_name : str or None
        Name of the embedding the current strata were built from.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Return the singleton ``RecoveryManager`` instance."""
        if not cls._instance:
            cls._instance = super(RecoveryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize (or reset) the recovery state to empty."""
        self.labels = None
        self.medoid_ids = None
        self.available_mask = None
        self.k_s = 0
        self.seed = 0
        self.n = 0
        self.active = False
        self.embedding_name = None
        self._segments = None
        self._medoid_labels = None

    def reset(self):
        """Clear all recovery state (called when the tractogram changes)."""
        self.__init__()

    @property
    def has_index(self):
        """Report whether strata usable for recovery are loaded.

        Distinct from :attr:`active`, which says only whether the *working set*
        was subsampled. Recovery also runs on tractograms below the subsampling
        threshold, where the index is built on demand by :meth:`ensure_index`.

        Returns
        -------
        bool
            True when fine strata and their medoids are available.
        """
        return (
            self.labels is not None
            and self.medoid_ids is not None
            and len(self.medoid_ids) > 0
        )

    def _invalidate_index_caches(self):
        """Drop derived views of the strata after ``labels`` changes."""
        self._segments = None
        self._medoid_labels = None

    def has_cached_labels(self, sft, embedding_name):
        """Report whether usable cached strata exist for ``embedding_name``.

        Used to decide up-front whether :meth:`build` will have to run the
        (slow) MiniBatchKMeans pass, so callers can show progress only when
        work is actually going to happen.

        Parameters
        ----------
        sft : StatefulTractogram
            The loaded tractogram.
        embedding_name : str
            The embedding whose cached strata are being looked for.

        Returns
        -------
        bool
            True when a valid cache is present and will be reused.
        """
        return self._read_cached_labels(sft, embedding_name) is not None

    def _read_cached_labels(self, sft, embedding_name):
        """Return validated cached strata for ``embedding_name``, else None.

        The cache lives in ``data_per_streamline``, which is arbitrary
        user-supplied data: an entry under the expected key may come from an
        unrelated pipeline. Labels are therefore only trusted when they are
        integral, non-negative, and bounded by the streamline count -- the
        assumptions :meth:`_subsample_from_labels` relies on. Anything else
        falls through to a fresh fine pass rather than silently steering it.

        Parameters
        ----------
        sft : StatefulTractogram
            The loaded tractogram.
        embedding_name : str
            The embedding whose cached strata are being read.

        Returns
        -------
        ndarray or None
            ``int32`` labels of shape ``(N,)``, or None when absent/unusable.
        """
        key = fine_labels_key(embedding_name)
        cached = sft.data_per_streamline.get(key, None)
        if cached is None:
            return None

        n = len(sft.streamlines)
        if n == 0:
            return None

        try:
            values = np.asarray(cached)
        except (ValueError, TypeError):
            logging.warning("Ignoring unreadable cached strata '%s'.", key)
            return None

        if values.size != n:
            logging.warning(
                "Ignoring cached strata '%s': %s values for %s streamlines.",
                key,
                values.size,
                n,
            )
            return None

        values = values.reshape(-1)
        if not np.issubdtype(values.dtype, np.number):
            logging.warning("Ignoring non-numeric cached strata '%s'.", key)
            return None
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            logging.warning("Ignoring non-integral cached strata '%s'.", key)
            return None

        labels = values.astype(np.int32)
        if labels.min() < 0 or labels.max() >= n:
            logging.warning(
                "Ignoring out-of-range cached strata '%s': ids span [%s, %s] "
                "for %s streamlines.",
                key,
                int(labels.min()),
                int(labels.max()),
                n,
            )
            return None

        return labels

    def build(self, sft, dismatrix, embedding_name):
        """Compute the working-set subsample and the recovery index.

        Runs once per tractogram. When the tractogram is at or below the
        subsampling threshold this is a no-op that returns every streamline id
        and leaves the manager inactive.

        Parameters
        ----------
        sft : StatefulTractogram
            The loaded tractogram; the fine-stratum labels are cached on its
            ``data_per_streamline`` under :func:`fine_labels_key` so they can
            persist with the tractogram and be reused without recomputation.
        dismatrix : ndarray
            The embedding of shape ``(N, num_prototypes)`` selected for
            clustering. Strata are built in this space, so the working set
            matches the space the clustering actually runs in.
        embedding_name : str
            Name of the embedding ``dismatrix`` was taken from. Qualifies the
            label cache so strata are never reused across embeddings.

        Returns
        -------
        ndarray
            ``int32`` streamline ids forming the interactive working set.
        """
        n = len(sft.streamlines)

        # Reuse cached labels (e.g. saved with the tractogram) when present.
        labels = self._read_cached_labels(sft, embedding_name)
        if labels is not None:
            result = self._subsample_from_labels(dismatrix, labels)
        else:
            result = stratified_subsample(dismatrix, seed=self.seed)

        sample_ids = np.asarray(result["sample_ids"], dtype=np.int32)
        self.labels = result["labels"]
        self.medoid_ids = result["medoid_ids"]
        self.k_s = result["k_s"]
        self.n = result["n"]
        self.active = self.labels is not None
        self.embedding_name = embedding_name
        self._invalidate_index_caches()

        if self.active:
            self.available_mask = np.ones(n, dtype=bool)
            self.available_mask[sample_ids] = False
            self._cache_labels(sft, embedding_name)
        else:
            self.available_mask = None

        logging.info(
            "Recovery index built on '%s': N=%s, active=%s, k_s=%s, working set=%s",
            embedding_name,
            n,
            self.active,
            self.k_s,
            len(sample_ids),
        )
        return sample_ids

    def _subsample_from_labels(self, dismatrix, labels):
        """Rebuild a subsample from cached fine-stratum labels.

        Mirrors the allocation in :func:`stratified_subsample` but skips the
        MiniBatchKMeans pass, reusing labels already cached on the tractogram.

        Parameters
        ----------
        dismatrix : ndarray
            The selected embedding, used only to recover medoids.
        labels : ndarray
            Cached ``int32`` stratum id for every streamline, already
            validated by :meth:`_read_cached_labels`.

        Returns
        -------
        dict
            Same shape as :func:`stratified_subsample`'s return value.
        """
        dismatrix = np.asarray(dismatrix, dtype=np.float32)
        n = labels.shape[0]
        pct = SUBSAMPLE_THRESHOLD / n
        rng = np.random.default_rng(self.seed)

        order, seg_starts, seg_ends = self._segment_labels(labels)
        n_clusters = len(seg_starts)
        medoid_ids = self._medoids_from_labels(dismatrix, order, seg_starts, seg_ends)
        medoid_by_stratum = dict(zip(labels[medoid_ids].tolist(), medoid_ids.tolist()))

        chosen = []
        for c in range(n_clusters):
            start, end = seg_starts[c], seg_ends[c]
            if start == end:
                continue
            members = order[start:end]
            medoid = medoid_by_stratum[c]

            size = end - start
            keep = min(size, max(KEEP_FLOOR, int(round(pct * size))))
            chosen.append(np.asarray([medoid], dtype=members.dtype))
            remaining = members[members != medoid]
            extra = min(keep - 1, len(remaining))
            if extra > 0:
                chosen.append(rng.choice(remaining, extra, replace=False))

        return {
            "sample_ids": np.unique(np.concatenate(chosen)).astype(np.int32),
            "labels": labels,
            "medoid_ids": medoid_ids,
            "k_s": n_clusters,
            "n": n,
        }

    @staticmethod
    def _segment_labels(labels):
        """Group streamline ids by stratum in one sort.

        Parameters
        ----------
        labels : ndarray
            Stratum id of every streamline.

        Returns
        -------
        order : ndarray
            Streamline ids sorted by stratum; stratum ``c`` owns the slice
            ``order[starts[c]:ends[c]]``.
        starts : ndarray
            Inclusive start offset of each stratum's slice.
        ends : ndarray
            Exclusive end offset of each stratum's slice.
        """
        order = np.argsort(labels, kind="stable").astype(np.int32)
        sorted_labels = labels[order]
        strata = np.arange(int(labels.max()) + 1)
        starts = np.searchsorted(sorted_labels, strata, side="left")
        ends = np.searchsorted(sorted_labels, strata, side="right")
        return order, starts, ends

    @staticmethod
    def _medoids_from_labels(dismatrix, order, seg_starts, seg_ends):
        """Find the member nearest each stratum's mean.

        Parameters
        ----------
        dismatrix : ndarray
            The selected embedding.
        order : ndarray
            Streamline ids sorted by stratum, from :meth:`_segment_labels`.
        seg_starts : ndarray
            Per-stratum start offsets into ``order``.
        seg_ends : ndarray
            Per-stratum end offsets into ``order``.

        Returns
        -------
        ndarray
            ``int32`` medoid streamline id per non-empty stratum, in ascending
            stratum order. Empty strata are omitted, so this array is *not*
            indexable by stratum id.
        """
        medoid_ids = []
        for c in range(len(seg_starts)):
            start, end = seg_starts[c], seg_ends[c]
            if start == end:
                continue
            members = order[start:end]
            diff = dismatrix[members] - dismatrix[members].mean(0)
            medoid_ids.append(int(members[np.einsum("ij,ij->i", diff, diff).argmin()]))
        return np.asarray(medoid_ids, dtype=np.int32)

    def _cache_labels(self, sft, embedding_name):
        """Store the current strata on the tractogram for later sessions.

        Parameters
        ----------
        sft : StatefulTractogram
            The tractogram to annotate.
        embedding_name : str
            The embedding the strata were built from.
        """
        key = fine_labels_key(embedding_name)
        try:
            sft.data_per_streamline[key] = self.labels.reshape(-1, 1)
        except Exception as exc:  # pragma: no cover - defensive cache only
            logging.warning("Could not cache '%s' on tractogram: %s", key, exc)

    def ensure_index(self, sft, dismatrix, embedding_name):
        """Guarantee that strata usable for recovery exist.

        :meth:`build` only produces strata when the tractogram was large enough
        to be subsampled. Recovery is useful below that threshold too -- fibers
        leave the scene through cluster deletion and ROI filtering, not only
        through subsampling -- so this builds the same index on demand, keeping
        recovery to a single code path at every tractogram size.

        Parameters
        ----------
        sft : StatefulTractogram
            The loaded tractogram; new strata are cached on it.
        dismatrix : ndarray
            The selected embedding of shape ``(N, num_prototypes)``.
        embedding_name : str
            Name of the embedding ``dismatrix`` was taken from. Strata built
            from a different embedding are discarded and rebuilt.

        Returns
        -------
        bool
            True when an index is available afterwards.
        """
        if self.has_index and self.embedding_name == embedding_name:
            return True

        dismatrix = np.asarray(dismatrix, dtype=np.float32)
        n = dismatrix.shape[0]
        if n == 0:
            return False

        labels = self._read_cached_labels(sft, embedding_name)
        if labels is None:
            # Below the subsampling threshold k_s scales with N so strata stay
            # around FINE_Q members each, the granularity recovery expects.
            k_s = max(1, min(n // FINE_Q, SUBSAMPLE_THRESHOLD // FINE_Q))
            labels, medoid_ids = fine_cluster_labels(dismatrix, k_s, seed=self.seed)
        else:
            order, seg_starts, seg_ends = self._segment_labels(labels)
            medoid_ids = self._medoids_from_labels(
                dismatrix, order, seg_starts, seg_ends
            )

        self.labels = labels
        self.medoid_ids = medoid_ids
        self.k_s = int(labels.max()) + 1
        self.n = n
        self.embedding_name = embedding_name
        self._invalidate_index_caches()
        self._cache_labels(sft, embedding_name)

        logging.info(
            "Recovery index ready on '%s': N=%s, strata=%s",
            embedding_name,
            n,
            len(medoid_ids),
        )
        return self.has_index

    def _stratum_segments(self):
        """Return the cached ``(order, starts, ends)`` grouping of the strata.

        Returns
        -------
        tuple
            The value of :meth:`_segment_labels` for the current labels.
        """
        if self._segments is None:
            self._segments = self._segment_labels(self.labels)
        return self._segments

    @property
    def medoid_labels(self):
        """Return the stratum id of each entry of :attr:`medoid_ids`.

        ``medoid_ids`` skips empty strata, so it cannot be indexed by stratum
        id directly. This ascending companion array makes the mapping in both
        directions cheap.

        Returns
        -------
        ndarray
            ``int32`` stratum ids, ascending, one per medoid.
        """
        if self._medoid_labels is None:
            self._medoid_labels = self.labels[self.medoid_ids].astype(np.int32)
        return self._medoid_labels

    def probe_strata(self, dismatrix, bundle_ids, *, nprobe=RECOVERY_NPROBE):
        """List the strata to search when growing ``bundle_ids``.

        The strata the bundle already occupies are the obvious candidates, but
        a bundle sitting near a stratum boundary has true neighbours just over
        it. Adding the ``nprobe`` nearest strata by medoid distance recovers
        those without widening the search to the whole tractogram.

        Parameters
        ----------
        dismatrix : ndarray
            The selected embedding of shape ``(N, num_prototypes)``.
        bundle_ids : array_like
            Streamline ids of the bundle being grown.
        nprobe : int, optional
            Extra strata probed per occupied stratum. ``0`` restricts the
            search to the occupied strata alone.

        Returns
        -------
        ndarray or None
            Ascending ``int32`` stratum ids to search, or None when no index is
            loaded (the caller should then fall back to the full pool).
        """
        if not self.has_index:
            return None

        bundle_ids = np.asarray(bundle_ids, dtype=np.int32).reshape(-1)
        if bundle_ids.size == 0:
            return None

        occupied = np.unique(self.labels[bundle_ids]).astype(np.int32)
        if nprobe <= 0:
            return occupied

        # Locate each occupied stratum among the (ascending) medoid labels.
        medoid_labels = self.medoid_labels
        positions = np.searchsorted(medoid_labels, occupied)
        valid = positions < medoid_labels.size
        positions = positions[valid]
        positions = positions[medoid_labels[positions] == occupied[valid]]
        if positions.size == 0:
            return occupied

        medoid_embedding = np.asarray(dismatrix, dtype=np.float32)[self.medoid_ids]
        neighbours = knn_indices(
            medoid_embedding, medoid_embedding[positions], int(nprobe) + 1
        )
        return np.union1d(occupied, medoid_labels[neighbours.reshape(-1)]).astype(
            np.int32
        )

    def stratum_members(self, strata):
        """Return every streamline id belonging to the given strata.

        Parameters
        ----------
        strata : array_like
            Stratum ids to gather.

        Returns
        -------
        ndarray
            ``int32`` streamline ids, unordered.
        """
        if not self.has_index:
            return np.empty(0, dtype=np.int32)

        order, seg_starts, seg_ends = self._stratum_segments()
        parts = []
        for stratum in np.asarray(strata, dtype=np.int64).reshape(-1):
            if not 0 <= stratum < len(seg_starts):
                continue
            start, end = seg_starts[stratum], seg_ends[stratum]
            if start < end:
                parts.append(order[start:end])
        if not parts:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(parts).astype(np.int32)

    def candidate_ids(self, *, exclude_ids, allowed_ids=None, strata=None):
        """Return the streamline ids eligible to be recovered.

        The pool is derived from the live scene rather than from
        :attr:`available_mask`: the mask is monotonic across a session, which
        would leave undone recoveries permanently excluded once the user steps
        back through the state history.

        Parameters
        ----------
        exclude_ids : array_like
            Streamline ids already on screen. Never returned.
        allowed_ids : array_like or None, optional
            When given, restricts the pool to these ids -- used to honour the
            active ROI filter so recovery cannot smuggle in fibers the filter
            excluded.
        strata : array_like or None, optional
            When given, restricts the pool to these strata (the coarse probe
            step). None searches every streamline.

        Returns
        -------
        ndarray
            ``int32`` candidate streamline ids.
        """
        n = int(self.n)
        if n <= 0:
            return np.empty(0, dtype=np.int32)

        mask = np.ones(n, dtype=bool)
        exclude = np.asarray(exclude_ids, dtype=np.int32).reshape(-1)
        if exclude.size:
            mask[exclude] = False

        if allowed_ids is not None:
            allowed = np.asarray(allowed_ids, dtype=np.int32).reshape(-1)
            keep = np.zeros(n, dtype=bool)
            if allowed.size:
                keep[allowed] = True
            mask &= keep

        if strata is None or not self.has_index:
            return np.flatnonzero(mask).astype(np.int32)

        members = self.stratum_members(strata)
        return members[mask[members]]


recovery_manager = RecoveryManager()
