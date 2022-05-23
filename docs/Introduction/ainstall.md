---
sort: 1
---

# Install and Build

DySim的下载只需要一条git命令即可完成：

```shell
git clone --recursive https://github.com/DyllanElliia/dySim.git
```

除外，还有部分依赖环境需要你本地配置一下：

|  库名  | 最小版本 | 是否必须 |
| :----: | :------: | :------: |
| OpenGL |   4.0    |    是    |
| OpenMP |   最新   |    是    |
|  TBB   |   最新   |    是    |
|  CUDA  |    11    |    否    |

> 该表可能存在更新不及时的问题，重点看CMake的报错提示哦~

可以使用如下命令开始编译：

```shell
mkdir build
cd build
cmake ..
make -j8
```

> `-j8`是`make`指令的一个参数，它可以让make多线程并行编译，能节省你不少时间~

文件结构上，cmake会将`/src`文件内的所有`*_main.c`、`*_main.cpp`和`*_main.cu`文件分别进行编译，输出对应的编译结果`*_main.out`或`*_main.exe`，保存在`/src`中。

若没有问题，当你运行编译结果`rtInOneWeek_main.out`时，是可以看到窗口显示一个带焦距的小球世界，你的CPU也会被这个程序吃满。

![rt_test_42](C:\code\cpp\dySim\src\rt_out\rtOneWeek\rt_test_42.jpg)
