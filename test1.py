# 迭代器，生成器

class Fab(object):
    def __init__(self,max):
        self.max = max
        self.n ,self.a,self.b = 0 , 0, 1
    def __iter__(self):
        return self
    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a ,self.b  = self.b ,self.a + self.b
            self.n = self.n + 1
            return  r
        raise StopIteration()

for n in Fab(5):
    print(n)


# 使用yield自动创建一个迭代器
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b  # 使用yield
        a, b = b, a+b
        n = n+1
for n in fab(5):
    print(n)


# 执行函数时，遇到yield就会中止操作，下次执行函数时从yield后面执行，知道遇到yield停止运行