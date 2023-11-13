a = ["a","b","c","d"]
aa = ["aa","bb","cc","dd"]

for a1,a2 in zip(a,aa):
    print(a1,a2)

print(" ")

b = "abcd"
bb = "ABCD"

for i, j in zip(b,bb):
    print(i,j)

print(" ")

for i, j in enumerate(b):
    print(i,j)

x = 5
cycle = list(range(-x, 0)) + list(range(1, x+1))
for i in cycle:
    print(i)

dĺžeň = "ja som dlzen"
print(dĺžeň)