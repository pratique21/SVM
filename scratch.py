from sklearn import svm

from benchmark import Benchmark
from visualize import Visualize

a = Benchmark.generate_polynomial(100, 0.001, 1)
b = Benchmark.generate_polynomial(100, 0.01, 6)
c = Benchmark.generate_ellipse(100, 0.0001)
d = Benchmark.generate_rectangle(100, 0.001)

for td in [a, b, c, d]:
    clf = svm.SVC()
    clf.fit(td[0], td[1])
    Visualize.plot_2d(td[0], td[1], td[2], show=False)
    Visualize.plot_clf(td[0], clf, show=True)
