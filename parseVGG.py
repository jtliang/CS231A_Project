import matplotlib.pyplot as plt
import numpy as np

vggFile = open("vgg16.txt")

x1s, x2s, baseline, noRotate, rotate, vgg = [x for x in range(1, 76)], [x for x in range(1, 34)], [], [], [], []

for line in vggFile:

  if "2155/2155" in line:
    acc = line.find("acc")
    val_loss = line.find("val_loss")
    acc = line[acc:val_loss]
    acc = float(acc.split(" ")[1])
    baseline.append(acc)

    val_acc = line.find("val_acc")
    vacc = line[val_acc:].strip()
    vacc = float(vacc.split(" ")[1])
    vgg.append(vacc)

baselineLegend,  = plt.plot(x2s, baseline, label='Training Accuracy')
noRotateLegend,  = plt.plot(x2s, vgg, label='Validation Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(handles=[baselineLegend, noRotateLegend])
plt.legend(bbox_to_anchor=(1, 0.3))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('VGG-16')
plt.show()