---
layout: home
---

![logoAddWord](index.assets/logoAddWord-16435318604501.png)

>Make Simulation Great Again🤺

## DySim 是什么🏷️

DySim 的名字取自于我的昵称 Dyllan 和这个工具 Simulator，表意就是“我的模拟器”！它起源于我的数学框架 DyMath，是对 DyMath 的拓展，因此你会看到非常多原框架遗留下的内容（如 `namespace dym`）。但由于 DyMath 这个库已经被别人创建了，所以我就将它命名为 DySim。

DySim 是什么？它是一个提供开发者更专注于写模拟的图形学架构，你可以用它很方便地验证一些数学方法。相比于其他方法，DySim 更适合：
1. 喜欢泛型和函数式编程的你🥰
2. 调各种材质参数的你😜
3. 懒得写各种并行的你🤔
4. 喜欢写CPP的你😂（符合这点的应该没多少人）

DySim 是我的本科毕业设计作品，它包含了泛型并行数学库、基于物质点法的物理模拟模块、基于光线追踪的渲染模块 和 一个简单的 GUI 模块。它或许不是一个能传世的框架，但它是我设计的适合我的一个 Simulator。若你喜欢这个 idea，给我一个 ⭐️ ！同时，欢迎一起让它变得更好！

## DySim总览

在 DySim 中，不同模块有不同类与方法，下表为你可在详细文档中所看到的内容：

|       类名       | 所属模块  | 功能                     |
| :--------------: | :-------: | :----------------------- |
|     `Vector`     |  dyMath   | 泛型向量计算             |
|     `Matrix`     |  dyMath   | 泛型矩阵计算             |
|     `Tensor`     |  dyMath   | 可用于并行计算的泛型模块 |
|     `Index`      |  dyMath   | 可用于`Tensor`的元素索引 |
|   `algorithm`    |  dyMath   | 上述类的数学方法部分     |
| `imread/imwrite` | dyPicture | 图像读写                 |
|  `picAlgorithm`  | dyPicture | 图像相关算法部分         |
|      `GUI`       |  dyGraph  | 简易 GUI 模块            |



## License

This work is open sourced under the Apache License, Version 2.0.

Copyright 2019 Tao He.

[1]: https://pages.github.com
[2]: https://pages.github.com/themes
[3]: https://github.com/sighingnow/jekyll-gitbook/fork
