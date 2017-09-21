from benchmark import Benchmark
from shapely.geometry import LineString, Point, Polygon
from visualize import Visualize

a = Benchmark.generate_linear(100, 0.001, 2)
Visualize.plot_2d(a[0], a[1], a[2], show=True)

b = Benchmark.generate_polynomial(100, 0.01, 2)
Visualize.plot_2d(b[0], b[1], b[2], show=True)

c = Benchmark.generate_ellipse(100, 0.0001)
Visualize.plot_2d(c[0], c[1], c[2], show=True)

d = Benchmark.generate_rectangle(100, 0.001)
Visualize.plot_2d(d[0], d[1], d[2], show=True)
