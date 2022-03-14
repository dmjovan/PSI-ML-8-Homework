def func(array):
    result = 0
    for i in range(0, len(array), 2):
        result += array[i]
    return result

print(func([1,2,3,4,5,6,7]))