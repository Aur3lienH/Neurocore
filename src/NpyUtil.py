import numpy
import os

dirname = os.path.dirname(__file__)
datasetPath = os.path.join(dirname, "../datasets/Quickdraw/")

for filename in os.listdir(datasetPath):
    if filename.endswith(".npy"):
        print(filename)
        data = numpy.load(datasetPath + filename)

        out = open(os.path.join(dirname, "../datasets/Quickdraw/custom_bin_" + filename[:-4] + ".bin"), "wb")
        i = 0
        binimgs = []
        for img in data:
            binimgs.append(img.tobytes())
            i += 1

        out.write(i.to_bytes(4, "little"))
        print("Drawings: ", i)

        for binimg in binimgs:
            out.write(binimg)

        out.close()
