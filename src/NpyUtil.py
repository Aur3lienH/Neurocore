import numpy
import os
import struct

dirname = os.path.dirname(__file__)

data = numpy.load(os.path.join(dirname, "../datasets/Quickdraw/full_numpy_bitmap_horse.npy"))

out = open(os.path.join(dirname, "../datasets/Quickdraw/custom_bin_horse.bin"), "wb")
i = 0
binimgs = []
for img in data:
    binimgs.append(img.tobytes())
    i+=1

out.write(i.to_bytes(4, "little"))
print(i)

for binimg in binimgs:
    out.write(binimg)

out.close()