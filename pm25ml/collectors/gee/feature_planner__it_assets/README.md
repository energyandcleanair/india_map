These have been manually generated for integration tests.

Each image covers 10km with 500m pixels. Each has 10 "spectral" bands and 1
category band.

The category band splits the 20x20 grid into a 4x4 grid of 5x5 pixels, with the
values increasing left to right and then top to bottom. The increasing values
continue from day to day, with 2022-12-30 having 0-15, 2022-12-31 having 16-31.
For example, day 1's 4x4 grid looks like this:
```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

The grid file partially covers the images with a roughly equal margin on each
side. The grid IDs are in the order:
```
2 3
0 1
```
