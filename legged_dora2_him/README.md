## 编译

```
cd ~/dora2/legged_dora2_him/
rm -r .catkin_tools build devel logs
source ~/.bashrc
conda deactivate
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
```

## 运行

启动仿真并加载控制器

```
./simulation.sh
```

or 

```
source ~/.bashrc
conda deactivate
./simulation.sh
```