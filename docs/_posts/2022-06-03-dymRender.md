---
title: DyRender 渲染器
author: DyllanElliia
date: 2022-06-03
category: Jekyll
layout: post
---

# 软件渲染器

当前的`dyRender`提供了基于软管线的光线追踪算法

## 通用API

![渲染类](./pic/渲染类.png)

渲染模块采用面向对象设计，基于三个基类衍生实现，使用智能指针相互关联。

> 全面采用智能指针，请尽可能使用`std::shared_ptr`！

### 贴图

贴图的基类是`Texture`，当前提供的基类仅两种，分别是纯色贴图`SolidColor`和纹理贴图`ImageTexture`。

~~~cpp
auto white_texture =
     std::make_shared<dym::rt::SolidColor>(dym::rt::ColorRGB(0.8f));

auto earth_texture =
     std::make_shared<dym::rt::ImageTexture>("image/earthmap.jpg");
~~~

> `ColorRGB`是一个颜色存储结构，是`Vector`的特化，因此它的使用方式也和`Vector`几乎一致。

### 材质

材质的基类是`Material`，若你先实现一个你先要的材质，你需要明确你要实现的材质的散射模型和着色模型，继承这个基类后，分别在`scatter`和`emitted`函数中进行实现。

|材质名|初始化参数|
|:---:|:---|
|`Dielectric`|`(电介质系数)`, `(ColorRGB,电介质系数)`, `(shared_ptr<Texture>,电介质系数)`|
|`DiffuseLight`|`(shared_ptr<Texture>)`, `(ColorRGB)`|
|`Lambertian`|`(ColorRGB)`, `(shared_ptr<Texture>)`|
|`Metal`|`(ColorRGB)`, `(shared_ptr<Texture>)`|

根据前面的信息，我们可以写一些材质：

~~~cpp
_DYM_FORCE_INLINE_ auto earthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture>("image/earthmap.jpg");
  auto earth_surface = std::make_shared<dym::rt::Lambertian>(earth_texture);

  return earth_surface;
}
_DYM_FORCE_INLINE_ auto whiteSur() {
  auto white_texture =
      std::make_shared<dym::rt::SolidColor>(dym::rt::ColorRGB(0.8f));
  auto white_surface = std::make_shared<dym::rt::Lambertian>(white_texture);

  return white_surface;
}
_DYM_FORCE_INLINE_ auto whiteMetalSur(Real fuzz = 0) {
  auto white_surface =
      std::make_shared<dym::rt::Metal>(dym::rt::ColorRGB(0.8f), fuzz);

  return white_surface;
}
_DYM_FORCE_INLINE_ auto whiteGalssSur() {
  auto white_surface = std::make_shared<dym::rt::Dielectric>(1.5);

  return white_surface;
}
_DYM_FORCE_INLINE_ auto lightEarthSur() {
  auto earth_texture =
      std::make_shared<dym::rt::ImageTexture>("image/earthmap.jpg", 3);
  auto earth_surface = std::make_shared<dym::rt::DiffuseLight>(earth_texture);

  return earth_surface;
}
~~~

> `_DYM_FORCE_INLINE_`是一个强制内联修饰，不确定这个代码是否要内联时，推荐使用`inline`修饰。

### 模型

模型的基类是`Hittable`


### BVH树

### 记录仪

### 模型加载器

#### 模型网格

## PTRender 基于重要性采样的路径追踪

### 案例一：Balls‘ world

### 案例二：Cornell Box

### 案例三：Moving Cornell Box

### 案例四：bunny Box
