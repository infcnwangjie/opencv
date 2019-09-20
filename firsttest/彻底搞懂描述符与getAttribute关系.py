'''定义一个描述符'''


class Desc:
    def __init__(self, value):
        self.value = value

    def __get__(self, instance, owner):
        print(owner)
        if instance:
            print("描述符作为属性被调用，并被调用get方法")
            return self.value
        else:
            print("描述符自己调用get方法")

    def __set__(self, instance, value):
        if instance:
            print("描述符被作为属性被调用set方法{0}".format(value))
            # 在这里应该做些类型校验，这个例子主要看调用关系，暂时省略
            self.value = value
        else:
            print("描述符自己调用set方法,{0}".format(value))
            self.value = value

    def __del__(self):
        print("描述符delete方法被调用")


class Animal:
    def __init__(self, age):
        self.age = Desc(age)


def first_test():
    dog = Animal(10)
    dog.age = 11
    print(dog.age)
    # 打印如下内容
    # C:\ProgramData\Anaconda3\envs\opencv\python.exe
    # E: / gitcodes / opencv / firsttest / 彻底搞懂描述符与getAttribute关系.py
    # 描述符delete方法被调用
    # 11


class AnimalSecond:
    '''描述符 必须定义为类属性,为了验证默认的getattribute方法调用描述符的get方法；
    　这是怎么实现的呢?要解释清楚这个原理,要先说明一下__getattribute__函数,当我们调用一个属性的时候,底层其实就是在执行该函数,该函数的工作方式是:
　　  B.x => B.__dict__['x'] => 如果 存在__get__方法 则 B.__dict__['x'].__get__(None, B)
    '''
    age = Desc(3)

    def __init__(self, age):
        self.age = age


def second_test():
    dog = AnimalSecond(7)
    dog.age = 11
    print(dog.age)
    # 打印如下内容
    # 描述符被作为属性被调用set方法7
    # 描述符被作为属性被调用set方法11
    # 描述符作为属性被调用，并被调用get方法
    # 11
    # 描述符delete方法被调用


class AnimalThird(object):
    '''要想覆盖getattribute方法，必须继承object类，否则getattribute会被无限调用；
    为了验证，getattribute方法调用描述符的规则：
    这是怎么实现的呢?要解释清楚这个原理,要先说明一下__getattribute__函数,当我们调用一个属性的时候,底层其实就是在执行该函数,该函数的工作方式是:
　　B.x => B.__dict__['x'] => 如果 存在__get__方法 则 B.__dict__['x'].__get__(None, B)'''
    age = Desc(3)

    def __init__(self, age):
        self.age = age

    def __getattribute__(self, item):
        print("调用getattribute,{0}".format(item))
        v = super().__getattribute__(item)
        # 使用super和使用object等价的无非，使用super不需要手动写self
        # v = object.__getattribute__(self, item)
        # 经过第四次验证，下面直接调用v.get也是多余的
        if hasattr(v, '__get__'):
            print("{0} has get".format(v))
            return v.__get__(None, self)
        return v


def third_test():
    dog = AnimalThird(7)
    dog.age = 11
    print(dog.age)
    # 打印如下内容
    # 描述符被作为属性被调用set方法7
    # 描述符被作为属性被调用set方法11
    # 调用getattribute, age
    # 描述符作为属性被调用，并被调用get方法
    # 11
    # 描述符delete方法被调用
    # 描述符delete方法被调用


class AnimalForth(object):
    '''要想覆盖getattribute方法，必须继承object类，否则getattribute会被无限调用；
    为了验证，getattribute方法调用描述符的规则，覆盖getattribut方法并且不提供age描述符的调用
    '''
    age = Desc(3)

    def __init__(self, age):
        self.age = age

    def __getattribute__(self, item):
        print("调用getattribute,{0}".format(item))
        v = super().__getattribute__(item)
        print("直接返回描述符也会调用get")
        return v


def forth_test():
    '''验证了只要调用父类的getattribute方法就会默认调用attribute'''
    dog = AnimalForth(7)
    dog.age = 11
    print(dog.age)
    # 打印如下内容
    # 描述符被作为属性被调用set方法7
    # 描述符被作为属性被调用set方法11
    # 调用getattribute, age
    # 描述符作为属性被调用，并被调用get方法
    # 11
    # 描述符delete方法被调用
    # 描述符delete方法被调用
    # 描述符delete方法被调用


class AnimalFifth(object):
    '''验证 get 和getattribute方法同时在一个类中会咋样，
    验证getattribute方法会被调用，而get方法除非指定调用否则不会被调用
    '''

    def __init__(self, age):
        self.age = age

    def __get__(self, instance, owner):
        print("描述符get调用")
        return self.age

    # def __getattribute__(self, item):
    #     print("调用getattribute,{0}".format(item))
    #     v = super().__getattribute__(item)
    #     return v


def fifth_test():
    '''验证了只要调用父类的getattribute方法就会默认调用attribute'''
    dog = AnimalFifth(7)
    dog.age = 11
    print(dog.age)
    # 打印如下内容:看出描述符没有被getattribute方法调用
    # 11
    # 描述符delete方法被调用
    # 描述符delete方法被调用
    # 描述符delete方法被调用


class AnimalSix:
    '''描述符 必须定义为类属性,为了验证默认的getattribute方法调用描述符的get方法；
    　这是怎么实现的呢?要解释清楚这个原理,要先说明一下__getattribute__函数,当我们调用一个属性的时候,底层其实就是在执行该函数,该函数的工作方式是:
　　  B.x => B.__dict__['x'] => 如果 存在__get__方法 则 B.__dict__['x'].__get__(None, B)
    '''
    age = Desc(3)

    def __init__(self, age):
        self.age = age

    def __getattribute__(self, item):
        print("打印AnimalSix的属性字典")
        print(AnimalSix.__dict__)
        # print("打印AnimalSix实例的属性字典")
        # print(hasattr(self,'__dict__'))#RecursionError: maximum recursion depth exceeded while getting the repr of an object

        obj = AnimalSix.__dict__.get(item)
        if hasattr(obj, '__get__'):
            print("调用Desc的get方法")
            return obj.__get__(self, AnimalSix)
        return self.age


def six_test():
    dog = AnimalSix(7)
    dog.age = 11
    print(dog.age)
    # 打印如下内容
    # 描述符被作为属性被调用set方法7
    # 描述符被作为属性被调用set方法11
    # 描述符作为属性被调用，并被调用get方法
    # 11
    # 描述符delete方法被调用


if __name__ == '__main__':
    # 描述符放到了__init__方法中定义
    # first_test()
    # 描述符作为类属性被定义
    # second_test()
    # 探索描述符还有getattribute方法调用关系
    # third_test()
    # 探索如果覆盖getattribute方法并且不提供描述符调用会咋样
    # forth_test()
    # 最终验证了一个问题，调用animal.age 首先会调用animal的getattribute方法，然后animal的getattribute方法，
    # 如果getattribute方法不被覆盖，那么默认会查找obj=Animal.__dict__.get('age')，然后判断age实例变量是否具有get方法
    # 如果具有get方法，就调用obj.__get__(None,self),如果覆盖了animal的getattribute，就看是否调用了super().getattribute(key),
    # 如果是，则仍会默认调用age实例变量的get方法，如果没有则错误
    # fifth_test()
    six_test()
