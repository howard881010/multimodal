import sys, statistics; print(statistics.median(map(float, open(sys.argv[1]).read().strip().split(','))))
