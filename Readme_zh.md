# PH-NET
[英文版本](Readme.md)

本仓库包含了论文 [PH-Net: Parallelepiped Microstructure Homogenization via 3D Convolutional Neural Networks](https://doi.org/10.1016/j.addma.2022.103237) 的代码。

您可以在下面找到有关训练自己的模型和使用预训练模型的详细使用说明。
如果您觉得我们的代码或论文有用，请引用：

    @article{peng2022ph,
        title={PH-Net: Parallelepiped microstructure homogenization via 3D Convolutional Neural Networks},
        author={Peng, Hao and Liu, An and Huang, Jingcheng and Cao, Lingxin and Liu, Jikai and Lu, Lin},
        journal={Additive Manufacturing},
        volume={60},
        pages={103237},
        year={2022},
        publisher={Elsevier}
    }

## 摘要
由于增材制造的快速发展，微结构引起了学术界和工业界的兴趣。数值均质化方法已经被广泛研究用于分析微结构的机械行为；然而，它在在线计算或需要高频调用时非常耗时，例如在拓扑优化中使用。数据驱动的均质化方法在求解中会更加有效。但是微结构仅限于立方体形状，因此不适用于具有更一般形状的周期性微结构，例如平行六面体。本文介绍了一种用于快速均质化平行六面体微结构的3D卷积神经网络（CNN），名为PH-Net。相对于现有的数据驱动方法，PH-Net预测了在指定宏观应变下微观结构的局部位移而不是直接预测均质材料，因此我们可以提出基于最小势能的无标签损失函数。为了构建数据集，我们引入了形状-材料转换和体素-材料张量来将微结构类型、基材料和边界形状编码为PH-Net的输入，使其适合于CNN，并增强了PH-Net在微结构类型、基材料和边界形状的泛化能力。PH-Net比数值均质化预测均质化特性快数百倍，甚至支持在线计算。此外，它不需要标记的数据集，因此训练过程比当前的深度学习方法要快得多。由于它可以预测局部位移，PH-Net提供了均质材料性质和微观机械性质，例如应变和应力分布以及屈服强度。我们还设计了一组使用3D打印材料的物理实验，以验证PH-Net的预测精度。

## 网络结构
<div align=center> 
    <img src="fig/net.jpg" width = 80%/> 
</div>

+ (a) 预处理将微结构、基材料和边界形状编码为材料-体素张量作为输入。
+ (b) 通过卷积神经网络预测微观位移，然后预测均质材料特性。
+ (c) 后处理步骤，恢复平行六面体微结构的均质材料。

## 使用
当您安装了所有依赖项并获得了预处理的数据后，您可以运行我们的预训练模型并从头开始训练新模型。

### 安装
您必须确保已经安装了所有依赖项以及第三方库，最简单的方法是使用：[anaconda](https://www.anaconda.com/). 

您可以使用以下命令创建名为`ph_net`的Anaconda环境：
```shell
conda env create -f environment.yaml
conda activate ph_net
```
### 数据生成
首先，我们需要一个可以使用体素表示的基础微结构模型。一旦有了这个模型，我们就可以使用基于体素的表示法生成各种仿射变换的模型作为训练数据。此外，我们还需要定义这些模型的基本属性，包括杨氏模量$E$和泊松比$v$。还需要计算模型的弹性张量$C^b$、刚度矩阵$K$和宏观力$f$，这些将作为`ph_net`的输入。

运行以下命令可以进行数据生成：
```sh
$ sh generate.sh
```
这个脚本的内容是：
```sh
CUDA_VISIBLE_DEVICES=0 python generate.py configs/tg.yaml
```
值得强调的是，在整个项目的配置中使用了`configs/tg.yaml`。在此文件中，`dataset`负责配置数据生成过程。
```yaml
dataset:
  out_dir: dataset/tg
  train_ratio: 0.8
  shuffle: true
  material:
    youngs_modulus_hard: 1
    youngs_modulus_soft: 1e-6
    poisson_ratio: 0.3
  voxel:
    dir: dataset/voxel/tg
    resolution: 36
    size: 40
  shape:
    sample_per_voxel: 1500
    scale_min: 1
    scale_max: 2
    angle_min: 75
    angle_max: 90
```
`material`部分定义了杨氏模量（`youngs_modulus_hard`和`youngs_modulus_soft`）和泊松比（`poisson_ratio`）。在`voxel`部分中，`dir`为基础微结构体素文件的目录。`resolution`定义了体素的分辨率，`size`是基础微结构数量。

在`shape`部分中，`sample_per_voxel`表示基于每个基础立方体微结构生成的平行六面体微结构数量。参数`scale_min`和`scale_max`定义了幅度尺度范围，而`angle_min`和`angle_max`表示坐标轴之间的角度范围。

### 训练
要从头开始训练一个新网络，请运行以下命令：
```shell
$ sh train.sh
```
这个脚本的内容是：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dp.py configs/tg.yaml --world_size=4
```
在 `configs/tg.yaml`中, `train` 部分用于配置 PH-Net 的训练.

```yaml
train:
  batch_size: 8
  out_dir: out/tg
  learning_rate: 5e-4
  epoch: 20
  eval_interval: 100
  pre_train: 'model.pt'
```

### 预测
通过将一组数据输入给网络，您可以直接获得大小为`(18*n*n*n)`的微观位移。弹性张量可以使用以下方法求解：
```python
C_homo=homo_net.homogenized(voxel,output,ke,X0)
``` 
在`network_homogenization`类中，可以使用`homogenize`方法求解弹性张量。该方法将输入体素、微观位移（output）、刚度矩阵（ke）和宏观应变（X0）作为输入，以均质化弹性张量，生成一个尺寸为`(36*36)`的张量。

<!-- Using `output = net(input)`, you can directly obtain the microscopic displacement of a set of input data $(18*n*n*n)$. The elasticity tensor is solved using `C_homo=homo_net.homogenized(voxel,output,ke,X0)` in class `network_homogenization`. Input voxel, microscopic displacement (output), stiffness matrix, macroscopic strain, can be solved to homogenize the elastic tensor$(36*36)$.  -->


## 样例
我们提供了一个基本数据集，它使用称为Tubular Gyroid（TG-TPMS）的三重周期性最小曲面作为基本微结构。该数据集包括40个均匀采样的体积分数，范围为 $[0.02, 0.33]$ 。对于每个体积分数，我们在一定的形状参数范围内选择了1500个不同的边界形状。幅度尺度范围设置为 $[1,2]$ ，而角度范围为 $[75^\circ, 90^\circ]$ ，总共产生了60k个样本。该数据集的弹性参数设置为$E_h = 1$ ，$v = 0.3$ ，$E_s = 1\times10^{-6}$ 。
下面的图片展示了我们提供的数据集，共包括40个模型。
<!-- We also provide a basic dataset. In this dataset, we choose a triply periodic minimal surface called Tubular Gyroid (TG-TPMS) as the basic microstructure with 40 uniform samples in the volume fraction $[2\%,33\%]$. 
We select 1500 distinct boundary shapes for each volume fraction sample in range of shape parameters.the magnitude scale range is $[1,2]$ and the angle range is $[75 ^\circ, 90^\circ]$, hence we have 60k samples in total. 
The other parameters are: $E_h = 1$,$v = 0.3$ and $E_s = 1×10^{−6}$. -->


<div align=center> 
    <img src="fig/model.jpg" width = 80%/> 
</div>

整个配置文件如下：
```yml
# configs/tg.yaml
dataset:
  out_dir: dataset/tg
  train_ratio: 0.8
  shuffle: true
  material:
    youngs_modulus_hard: 1
    youngs_modulus_soft: 1e-6
    poisson_ratio: 0.3
  voxel:
    dir: dataset/voxel/tg
    resolution: 36
    size: 40
  shape:
    sample_per_voxel: 1500
    scale_min: 1
    scale_max: 2
    angle_min: 75
    angle_max: 90
train:
  batch_size: 8
  out_dir: out/tg
  learning_rate: 5e-4
  epoch: 100
  eval_interval: 100
  pre_train: 'model.pt'
```
修改 `generate.sh` 的内容
```bash 
CUDA_VISIBLE_DEVICES=0 python generate.py configs/tg.yaml
```
修改 `train.sh` 的内容
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dp.py configs/tg.yaml --world_size=4
```
## 许可
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />本工作基于以下许可 <a rel="license" href="LICENSE">Creative Commons Attribution-NonCommercial 4.0 International License</a>.



<!-- <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/3.0/88x31.png" /></a>
This project is licensed under the [Creative Commons Attribution-NonCommercial (CC BY-NC) 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/deed.zh). This means that you are free to use, copy, distribute, and transmit the project for non-commercial purposes. -->