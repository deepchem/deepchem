class RunningSearchStatistics:
    def __init__(self, maxsize, window_size=100000):
        self.maxsize = maxsize
        self.window_size = window_size
        self.frequencies = [1.0 for _ in range(self.maxsize)]
        self._normalize_inplace()

    def copy(self):
        out = RunningSearchStatistics(self.maxsize, self.window_size)
        out.frequencies = list(self.frequencies)
        out.normalized_frequencies = list(self.normalized_frequencies)
        return out

    def update(self, size):
        if 1 <= size <= self.maxsize:
            self.frequencies[size - 1] += 1.0

    def update_many(self, sizes):
        for size in sizes:
            self.update(size)

    def move_window(self):
        """
        Reduce total frequency mass to `window_size`, while keeping each entry >= 1.
        """
        total = sum(self.frequencies)
        if total <= self.window_size:
            return

        smallest_allowed = 1.0
        diff = total - self.window_size
        loops = 0
        max_loops = 1000

        while diff > 0 and loops < max_loops:
            idx = [i for i, f in enumerate(self.frequencies) if f > smallest_allowed]
            if not idx:
                break

            min_above = min(self.frequencies[i] for i in idx) - smallest_allowed
            if min_above <= 0:
                break

            amount = min(diff / len(idx), min_above)
            if amount <= 0:
                break

            for i in idx:
                self.frequencies[i] -= amount
            diff -= amount * len(idx)
            loops += 1

    def normalize(self):
        self._normalize_inplace()

    def _normalize_inplace(self):
        total = sum(self.frequencies)
        if total <= 0:
            self.normalized_frequencies = [
                1.0 / max(1, self.maxsize) for _ in range(self.maxsize)
            ]
        else:
            self.normalized_frequencies = [f / total for f in self.frequencies]

    @staticmethod
    def merge(stats):
        stats_list = list(stats)
        if not stats_list:
            return None
        maxsize = stats_list[0].maxsize
        window_size = stats_list[0].window_size
        merged = RunningSearchStatistics(maxsize, window_size)
        merged.frequencies = [0.0 for _ in range(maxsize)]
        for st in stats_list:
            if st.maxsize != maxsize:
                raise ValueError(
                    "Cannot merge RunningSearchStatistics with different maxsize."
                )
            for i in range(maxsize):
                merged.frequencies[i] += st.frequencies[i]
        merged._normalize_inplace()
        return merged
