---
sort: 1
---

# Vector 向量

dyMath提供了绝大多数的数值计算所需的向量工具，包含常用的向量函数。

在使用dyMath前，你需要了解dyMath提供的替代`int`、`float`和`double`等变量的符号，而且非常推荐使用dyMath提供的替换符号：

|变量名|含义|
|:---:|:---:|
|`Real`|64位浮点|
|`Reali`|32位整形|
|`uReal`|32位无符号整型|

> 关于`Real`的选择，起初考虑过32位浮点，但在做高精度模拟时发现，这并不实用，而且本人做高精度模拟更多一些，因此选择默认64位浮点。

## 基本操作
### 创建一个Vector


~~~cpp
template <typename Type, std::size_t dim>
struct Vector;
~~~

`Vector`是一个模板，使用它前你需要定义它存储什么类型的变量和它的维度，按照初始化提供初始化内容，创建如下：

~~~cpp
dym::Vector<Real, 3> a(1);
Out: a = {1.0, 1.0, 1.0}
~~~

~~~cpp
dym::Vector<Real, 3> b({1, 2, 3});
Out: b = {1.0, 2.0, 3.0}
~~~

当输入为空，即`dym::Vector<Real, 3>()`，等效于`dym::Vector<Real, 3>(Real(0))`。

除了使用数字构造向量，还能使用向量构造向量：

~~~cpp
auto d = dym::Vector<Real, 5>(b, 6);
Out: d = {1.0, 2.0, 3.0, 6.0, 6.0, 6.0}
~~~

~~~cpp
auto e = dym::Vector<Real, 2>(d);
Out: e = {1.0, 2.0}
~~~

为了提供一些特殊使用一些帮助，`Vector`支持使用Lambda表达式进行构造：

~~~cpp
dym::Vector<Real, 10> g([&](Real& e, int i) { e = i; });
Out: g = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
~~~

### 向量操作符

dyMath对`Vector`提供了非常完整的操作符适配，基本满足大多数需求。

对于`=`，对常见操作提供了赋值操作，对于不同维度的`Vector`提供了统一赋值：

~~~cpp
dym::Vector<Real, 3> a;
a = 10.0;
Out: a = {10.0, 10.0, 10.0}
~~~

~~~cpp
dym::Vector<Real, 3> a;
dym::Vector<Real, 3> b({1, 2, 3});
a = b;
Out: a = {1.0, 2.0, 3.0}
~~~

~~~cpp
dym::Vector<Real, 2> c(1.0);
dym::Vector<Real, 4> d(1.0);
c = b, d = b;
Out: c = {1.0, 2.0}
     d = {1.0, 2.0, 3.0, 1.0}
~~~

对于不同变量类型的`Vector`赋值，dyMath提供了转换模板`cast`，使用如下：

~~~cpp
dym::Vector<Real, 3> b({1, 2, 3});
auto a = b.cast<Reali>();
Out: a = {1, 2, 3}
~~~

除外，还有常用数学操作符，首先为了简写，使用`R`代表实数`Real`和`Reali`等，使用`V`代表向量`Vector`：

|操作符|操作|备注|
|:---:|:---|:---|
|`+`|`R+V`, `V+R`, `V+V`||
|`-`|`R-V`, `V-R`, `V-V`||
|`*`|`R*V`, `V*R`||
|`/`|`R/V`, `V/R`||
|`+=`|`V+=R`, `V+=V`||
|`-=`|`V-=R`, `V-=V`||
|`*=`|`V*=R`, `V*=V`||
|`/=`|`V/=R`, `V/=V`||
|`<`|`V<R`, `V<V`|所有变量一一满足时返回`true`|
|`>`|`V>R`, `V>V`|同上|
|`<=`|`V<=R`, `V<=V`|同上|
|`>=`|`V>=R`, `V>=V`|同上|
|`==`|`V==R`, `V==V`|等效于`abs(a-b)<1e-7`|

> 对于`V/V`这个操作，数学定义上是不存在的，因为缺乏意义。但还是考虑到了可能存在的需求，保留了`V/=V`，定义为：
> 
> $$\cfrac {\vec a} {\vec b}=(\cfrac{\vec a_1}{\vec b_1},\cfrac{\vec a_2}{\vec b_2},...,\cfrac{\vec a_n}{\vec b_n})$$
> 
> 同样，将乘法中`V*V`做了同样的定义：
> 
> $$\vec a*\vec b=(\vec a_1*\vec b_1,\vec a_2*\vec b_2,...,\vec a_n*\vec b_n)$$
> 
> 其他乘法，如`dot`和`cross`，单独作为一个数值计算函数。

### 元素访问

dyMath提供两种方法元素访问：

~~~cpp
dym::Vector<Real, 3> a(1.0);
dym::Vector<Real, 3> b(2.0);
a[0] = 2; // 有效
b[1] = a[1];
Out: a = {2.0, 1.0, 1.0}
     b = {2.0, 1.0, 2.0}
~~~

~~~cpp
dym::Vector<Real, 3> a(1.0);
dym::Vector<Real, 3> b(2.0);
a.y() = 2; // 无效
b[1] = a.y();
Out: a = {1.0, 1.0, 1.0}
     b = {2.0, 1.0, 2.0}
~~~

> 使用函数进行访问，最多只支持到4维向量，即`x()`、`y()`、`z()`和`w()`。

## 向量数值方法

### dot & cross

实现两个向量的点乘与叉乘。

~~~cpp
dym::Vector<Real, 3> a(1.0);
dym::Vector<Real, 3> b(2.0);
Real c;

// dot
c = a.dot(b);
c = dym::vector::dot(a,b);

Out: c = 6.0
~~~

~~~cpp
dym::Vector<Real, 3> a(1.0);
dym::Vector<Real, 3> b(2.0);
dym::Vector<Real, 3> c;

// cross
c = a.cross(b);
c = dym::vector::cross(a,b);

Out: c = {0.0, 0.0, 0.0}
~~~

叉乘方法支持高维向量叉乘：

~~~cpp
dym::Vector<Real, 3> c;

// cross
c = dym::Vector<Real, 4>({ 1.0,  2.0, 3.0, 4.0}).cross(
    dym::Vector<Real, 4>({-4.0, -2.0, 3.0, 1.0}),
    dym::Vector<Real, 4>({10.0, -8.0, 6.0, 5.0}));

c = dym::vector::cross(dym::Vector<Real, 4>({ 1.0,  2.0, 3.0, 4.0}),
                       dym::Vector<Real, 4>({-4.0, -2.0, 3.0, 1.0}),
                       dym::Vector<Real, 4>({10.0, -8.0, 6.0, 5.0}));

Out: c = {72.0, 117.0, 266.0, -276.0}
~~~

### length & length_sqr 模长

实现计算向量模长：

~~~cpp
dym::Vector<Real, 2> a({3.0, 4.0});

Real b = a.length_sqr();

Out: b = 25.0
~~~

~~~cpp
dym::Vector<Real, 2> a({3.0, 4.0});
Real b;

b = dym::sqrt(a.length_sqr());
b = a.length();

Out: b = 5.0
~~~
### normalize 单位化

实现向量单位化为单位向量：

~~~cpp
dym::Vector<Real, 3> a({0.0, 3.0, 0.0});
dym::Vector<Real, 3> b;

b = a.normalize();
b = dym::vector::normalized(a);

Out: b = {0.0, 1.0, 0.0}
~~~

### reflect 反射

图形学中经常要算一个向量关于一个法线的反射：

$$\vec a'=\vec a-2(\vec a \cdot \vec normal)*\vec normal$$

因此，dyMath提供了`Vector`的反射函数：

~~~cpp
dym::Vector<Real, 3> a({1.0, -1.0, 0.0});
dym::Vector<Real, 3> normal({0.0, 1.0, 0.0});

auto b = a.reflect(normal);

Out: b = {1.0, 1.0, 0.0}
~~~