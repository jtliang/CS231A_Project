import sqlite3
from random import random

# conn = sqlite3.connect('train.db')
# c = conn.cursor()

# speciesToID = {}
# idCounter = 0

# f = open('train.csv')
# items = []
# c.execute('DROP TABLE IF EXISTS train')
# for i, line in enumerate(f):
#   lineArr = line.split(',')[:2]
#   # schema
#   if i == 0:
#     createSchema = 'CREATE TABLE train ('
#     for j in range(2):
#       item = lineArr[j]
#       if j > 0:
#         createSchema += ', '
#       createSchema += item
#     createSchema += ')'
#     c.execute(createSchema)
#   else:
#     for j in range(2):
#       item = lineArr[j]
#       if j == 0:
#         lineArr[j] = int(item)
#       elif j == 1:
#         species = item
#         if species not in speciesToID:
#           speciesToID[species] = idCounter
#           idCounter += 1
#         lineArr[j] = int(speciesToID[species])
#     asTuple = tuple(lineArr)
#     items.append(asTuple)
# c.executemany('INSERT INTO train VALUES (?, ?)', items)
# conn.commit()
# c.execute('DROP TABLE IF EXISTS test')
# conn.commit()
# conn.close()



conn = sqlite3.connect('train.db')
c = conn.cursor()
res = c.execute("SELECT id, species from train where id < 1585")
train_datas = []
train_labels = []
test_datas = []
test_labels = []
for r in res:
  prob = random()
  if prob >= 0.8:
    test_datas.append(str(r[0]))
    test_labels.append(str(r[1]))
  else:
    train_datas.append(str(r[0]))
    train_labels.append(str(r[1]))
training = open("train2.txt", 'a')
training.write(','.join(train_datas) + "\n")
training.write(','.join(train_labels) + "\n")
testing = open("test2.txt", 'a')
testing.write(','.join(test_datas) + "\n")
testing.write(','.join(test_labels) + "\n")
conn.close()
