"""Interactive loop generator (intersections only)

Click points on a canvas to define a polyline loop and auto-detect proper
segment-segment intersections (4-way crossings where two interior segments meet).

How it works
- Left-click to add points. A polyline is drawn between consecutive points.
- Finish & Build: closes the loop (last to first), finds proper crossings, marks
	them on the canvas, and prints a concise summary and per-segment crossing order.
- Undo Last: removes the last point. Clear: clears points and drawings.

Note: Angles and orientation-sensitive steps use math coordinates (y up), so
we invert the canvas y wherever needed.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Dict, List, Tuple, Optional


Point = Tuple[float, float]


def _round_key(p: Point, eps: float = 1e-6) -> Tuple[int, int]:
	return (int(round(p[0] / eps)), int(round(p[1] / eps)))


def _sub(a: Point, b: Point) -> Tuple[float, float]:
	return (a[0] - b[0], a[1] - b[1])


def _cross(u: Point, v: Point) -> float:
	return u[0] * v[1] - u[1] * v[0]


def _dot(u: Point, v: Point) -> float:
	return u[0] * v[0] + u[1] * v[1]


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
	# p colinear with a-b and within bounding box
	ab = _sub(b, a)
	ap = _sub(p, a)
	if abs(_cross(ab, ap)) > eps:
		return False
	minx, maxx = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
	miny, maxy = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
	return (minx - eps) <= p[0] <= (maxx + eps) and (miny - eps) <= p[1] <= (maxy + eps)


def segment_intersection(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> Optional[Tuple[float, float, float, float, Point]]:
	"""Compute intersection between segments ab and cd.

	Returns (t, u, ix, iy, (ix, iy)) where:
	  - point = a + t*(b-a) = c + u*(d-c)
	  - If no intersection, returns None.
	Includes endpoint intersections; caller can filter if needed.
	Handles colinear overlaps by returning None here; caller should handle via endpoint checks.
	"""
	r = _sub(b, a)
	s = _sub(d, c)
	rxs = _cross(r, s)
	q_p = _sub(c, a)
	qpxr = _cross(q_p, r)

	if abs(rxs) < eps and abs(qpxr) < eps:
		# Colinear - overlapping handled by caller with endpoint checks
		return None
	if abs(rxs) < eps and abs(qpxr) >= eps:
		# Parallel non-intersecting
		return None

	t = _cross(q_p, s) / rxs
	u = _cross(q_p, r) / rxs
	if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
		ix = a[0] + t * r[0]
		iy = a[1] + t * r[1]
		return (t, u, ix, iy, (ix, iy))
	return None


def project_t_on_segment(a: Point, b: Point, p: Point) -> float:
	# Return t in [0,1] such that a + t*(b-a) is projection of p
	ab = _sub(b, a)
	ab2 = _dot(ab, ab)
	if ab2 == 0:
		return 0.0
	t = _dot(_sub(p, a), ab) / ab2
	return max(0.0, min(1.0, t))


class HalfEdgeBuilder:
	def __init__(self, points: List[Point]) -> None:
		self.points = points
		# intersection metadata keyed by rounded point
		self.intersections: Dict[Tuple[int, int], Dict[str, object]] = {}
		# per-segment splits (filled in finder)
		self.seg_splits: List[List[Dict[str, object]]] = []
		self.cross_id_by_key: Dict[Tuple[int, int], int] = {}
		self.cross_occurrences: Dict[int, List[Tuple[int, int]]] = {}

	def find_intersections(self) -> Tuple[List[Dict[str, object]], Dict[int, List[Tuple[float, int, Point]]]]:
		"""Compute proper interior intersections and return:
		- intersections: list of {id, point, segments: (i,j)}
		- seg_to_crossings: mapping seg_idx -> list of (t, id, point) sorted by t
		"""
		if len(self.points) < 2:
			return [], {}
		# Build t-map and intersection metadata
		tmap = self._collect_intersections()
		segs = self._close_loop_segments()
		# Assign crossing ids only for proper intersections
		proper_keys = [key for key, meta in self.intersections.items() if meta.get("proper")]
		self.cross_id_by_key = {key: idx + 1 for idx, key in enumerate(sorted(proper_keys))}
		# Build per-segment splits and per-segment crossing order
		self.seg_splits = []
		seg_to_crossings: Dict[int, List[Tuple[float, int, Point]]] = {}
		self.cross_occurrences = {}
		for i in range(len(segs)):
			splits_i = []
			seg_crosses: List[Tuple[float, int, Point]] = []
			for (t, p) in tmap[i]:
				key = _round_key(p)
				cid = self.cross_id_by_key.get(key)
				is_cross = cid is not None
				ent = {"t": t, "pt": p, "key": key, "is_cross": is_cross, "cross_id": cid}
				splits_i.append(ent)
				if is_cross:
					seg_crosses.append((t, cid, p))
					self.cross_occurrences.setdefault(cid, []).append((i, len(splits_i) - 1))
			seg_to_crossings[i] = sorted(seg_crosses, key=lambda x: x[0])
			self.seg_splits.append(splits_i)
		# Build intersections list
		intersections: List[Dict[str, object]] = []
		for key, cid in self.cross_id_by_key.items():
			pt = None
			for i, splits in enumerate(self.seg_splits):
				for ent in splits:
					if ent.get("key") == key:
						pt = ent.get("pt")
						break
				if pt is not None:
					break
			segs_involved = set()
			for (seg_idx, _) in self.cross_occurrences.get(cid, []):
				segs_involved.add(seg_idx)
			if len(segs_involved) == 2 and pt is not None:
				intersections.append({
					"id": cid,
					"point": pt,
					"segments": tuple(sorted(segs_involved)),
				})
		# Sort intersections by id
		intersections.sort(key=lambda d: d["id"]) 
		return intersections, seg_to_crossings

	def _close_loop_segments(self) -> List[Tuple[Point, Point, int]]:
		segs = []
		n = len(self.points)
		for i in range(n):
			a = self.points[i]
			b = self.points[(i + 1) % n]
			segs.append((a, b, i))
		return segs

	def _collect_intersections(self) -> Dict[int, List[Tuple[float, Point]]]:
		"""For each segment index i, produce parameter values t in [0,1] to split.
		Always includes t=0 and t=1 for endpoints.
		"""
		segs = self._close_loop_segments()
		n = len(segs)
		tmap: Dict[int, List[Tuple[float, Point]]] = {i: [(0.0, segs[i][0]), (1.0, segs[i][1])] for i in range(n)}
		for i, (a, b, _) in enumerate(segs):
			for j in range(i + 1, n):
				c, d, _ = segs[j]
				# skip checking neighbors that share endpoints to avoid duplicate t; de-dup later anyway
				inter = segment_intersection(a, b, c, d)
				if inter is not None:
					t, u, ix, iy, p = inter
					# Add interior points; endpoints allowed, will de-dup
					tmap[i].append((max(0.0, min(1.0, t)), (ix, iy)))
					tmap[j].append((max(0.0, min(1.0, u)), (ix, iy)))
					# record proper interior crossing if both strictly inside
					if 0.0 < t < 1.0 and 0.0 < u < 1.0:
						key = _round_key((ix, iy))
						meta = self.intersections.setdefault(key, {"segments": set(), "proper": False})
						meta["segments"].update({i, j})
						meta["proper"] = True
				else:
					# Handle colinear overlapping by considering overlapping endpoints
					for p in (a, b, c, d):
						if _on_segment(a, b, p) and _on_segment(c, d, p):
							ti = project_t_on_segment(a, b, p)
							tj = project_t_on_segment(c, d, p)
							tmap[i].append((ti, p))
							tmap[j].append((tj, p))
							key = _round_key(p)
							meta = self.intersections.setdefault(key, {"segments": set(), "proper": False})
							meta["segments"].update({i, j})
		# de-duplicate and sort
		for i in range(n):
			seen = {}
			for t, p in tmap[i]:
				key = _round_key(p)
				seen[key] = (t, p)
			tmap[i] = sorted(seen.values(), key=lambda x: x[0])
		return tmap


 


class LoopUI:
	def __init__(self, width: int = 800, height: int = 600) -> None:
		self.root = tk.Tk()
		self.root.title("Generate Loop - Click to add points")
		self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
		self.canvas.pack(fill=tk.BOTH, expand=True)

		self.points: List[Point] = []
		self.point_handles: List[int] = []
		self.segment_handles: List[int] = []
		self.intersection_handles: List[int] = []

		btn_frame = tk.Frame(self.root)
		btn_frame.pack(fill=tk.X)
		tk.Button(btn_frame, text="Finish & Build", command=self.finish_and_build).pack(side=tk.LEFT)
		tk.Button(btn_frame, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT)
		tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

		self.canvas.bind("<Button-1>", self.on_click)

	def on_click(self, event: tk.Event) -> None:
		p = (float(event.x), float(event.y))
		self.points.append(p)
		self._draw_point(p, color="#1f77b4")
		n = len(self.points)
		if n >= 2:
			self._draw_segment(self.points[-2], self.points[-1], color="#888888")

	def _draw_point(self, p: Point, color: str = "red", r: int = 3) -> None:
		x, y = p
		h = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")
		self.point_handles.append(h)

	def _draw_segment(self, a: Point, b: Point, color: str = "black", w: int = 2) -> None:
		h = self.canvas.create_line(a[0], a[1], b[0], b[1], fill=color, width=w)
		self.segment_handles.append(h)

	def _draw_intersection(self, p: Point, color: str = "#d62728", r: int = 3) -> None:
		x, y = p
		h = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")
		self.intersection_handles.append(h)

	def undo_last(self) -> None:
		if not self.points:
			return
		self.points.pop()
		# remove last point handle
		if self.point_handles:
			self.canvas.delete(self.point_handles.pop())
		# remove last segment if any
		if self.segment_handles:
			self.canvas.delete(self.segment_handles.pop())

	def clear(self) -> None:
		self.points.clear()
		for lst in (self.point_handles, self.segment_handles, self.intersection_handles):
			while lst:
				self.canvas.delete(lst.pop())

	def finish_and_build(self) -> None:
		if len(self.points) < 3:
			print("Need at least 3 points to form a loop.")
			return
		# draw closing segment
		self._draw_segment(self.points[-1], self.points[0], color="#888888")

		# Identify intersections only (ignore half-edges)
		builder = HalfEdgeBuilder(self.points)
		intersections, seg_to_cross = builder.find_intersections()

		# Draw intersections and label with id
		for inter in intersections:
			pt = inter["point"]
			cid = inter["id"]
			self._draw_intersection(pt)
			# label
			x, y = pt
			self.canvas.create_text(x + 8, y - 8, text=str(cid), fill="#d62728", anchor="w")

		print("\n=== Intersections (proper 4-way) ===")
		print(f"Total: {len(intersections)}")
		for inter in intersections:
			cid = inter["id"]
			x, y = inter["point"]
			segs = inter["segments"]
			print(f"{cid}: point=({x:.2f}, {y:.2f}), segments={segs}")

		# Optional: per-segment ordering of crossings along the strand
		print("\nPer-segment crossing order (t from 0->1):")
		for seg_idx in sorted(seg_to_cross.keys()):
			arr = seg_to_cross[seg_idx]
			ids = [cid for (t, cid, p) in arr]
			print(f"segment {seg_idx}: {ids}")


def main() -> None:
	ui = LoopUI()
	ui.run()


if __name__ == "__main__":
	main()

