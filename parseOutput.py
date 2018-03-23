import matplotlib.pyplot as plt
import numpy as np

baselineFile = open("baseline.txt")
noRotateFile = open("outputNoRotate.txt")
rotateFile = open("output.txt")
vggFile = open("vgg16.txt")

x1s, x2s, baseline, noRotate, rotate, vgg = [x for x in range(1, 76)], [x for x in range(1, 34)], [], [], [], []

for line in baselineFile:
  if "724/724" in line:
    val_acc = line.find("val_acc")
    vacc = line[val_acc:].strip()
    vacc = float(vacc.split(" ")[1])
    baseline.append(vacc)

for line in noRotateFile:
  if "724/724" in line:
    val_acc = line.find("val_acc")
    vacc = line[val_acc:].strip()
    vacc = float(vacc.split(" ")[1])
    noRotate.append(vacc)

for line in rotateFile:
  if "2155/2155" in line:
    val_acc = line.find("val_acc")
    vacc = line[val_acc:].strip()
    vacc = float(vacc.split(" ")[1])
    rotate.append(vacc)

for line in vggFile:
  if "2155/2155" in line:
    val_acc = line.find("val_acc")
    vacc = line[val_acc:].strip()
    vacc = float(vacc.split(" ")[1])
    vgg.append(vacc)

baselineLegend,  = plt.plot(x1s, baseline, label='baseline')
noRotateLegend,  = plt.plot(x1s, noRotate, label='no rotation')
rotateLegend,  = plt.plot(x1s, rotate, label='rotation and edge')
vggLegend,  = plt.plot(x2s, vgg, label='vgg-16')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(handles=[baselineLegend, noRotateLegend, rotateLegend, vggLegend])
plt.legend(bbox_to_anchor=(1, 0.3))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Comparison')
plt.show()