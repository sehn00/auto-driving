import c_example
import time

def add(a, b): return a + b
start = time.time()
#for i in range(1000000): 
#    result = c_example.add_int(3,5)
for i in range(1000000):
    result = add(3, 5)
end = time.time()
print(result)
print("Elapsed time:", end - start)