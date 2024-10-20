# Alacrity
A host-device library. <br>
Works to incorporate with a single host device and a multitude of the same device type. <br>
<br>
The main library will be header mainly a header library with lots of templates. <br>
There will also be a runtime library but it will most likely be slower, <br>
if you have your own specific needs the runtime library is still tested and should be safe. <br>

### Build && Install
```bash
mkdir build/ && cd build/
cmake ..
cmake --build .
sudo cmake --install .
```

### Testing
```bash
cd unit_tests/
make # host
make cu # cuda tests
```

### Compatible Devices
CPU's (no specifics for x86 or arm but tested on an x86 machine) <br>
CUDA  (cuda compatible devices (i.e nvidia graphic cards)) <br>
HIP   (Comming Soon...) <br>

# Reference
[Tsoding nn.h](https://github.com/tsoding/nn.h) <br>

# TODO
Remove internal malloc in favour of dynamic buffer, <br>
using malloc and free multiple times in an otherwise asynchrounous function is sub-optimal. <br>
In fact using malloc at all is sub-optimal, to fix this consider setting a global arena buffer, <br>
this will take a bit more work but should overall be faster. <br>
<br>
Use Makefiles instead of cmake (less boilerplate). <br>
