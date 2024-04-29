# Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes

![image](img/teaser.png)

[Yue Li](https://liyuesolo.github.io/), Logan Numerow, [Bernhard Thomaszewski](https://n.ethz.ch/~bthomasz/), [Stelian Coros](https://crl.ethz.ch/people/coros/index.html)

ACM Transactions on Graphics (Proc. SIGGRAPH 2024)
### [Paper]() [Video]()

## Project Structure

Projects/DifferentiableGeodesics/
- [include & src] contains header and source files
- [autodiff] contains the derivative expressions generated by a compile time autodiff library
- [data] contains the meshes used in the paper
- [result] is the default output folder

## Compile and Run 
On Linux machines, the provided docker environment should work out of the box. On MacOS, cmake can also work once all required libraries are installed using homebrew.

### Install NVIDIA Docker

> $ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
> $ sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

> $ sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit

> $ nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

> $ sudo nvidia-ctk runtime configure --runtime=docker


### Run Docker in VSCode

Download the docker image (faster option, recommended). 
> $ docker pull wukongsim/differentiable_geodesics_environment:latest

If you wish to build the docker image from scratch from the Dockerfile see .devcontainer/.json for instructions (simply uncommenting two lines).

Open the repository folder in VSCode. Install Docker and Dev Containers extensions in VSCode.

In VSCode, type `control + p`, then type `>Reopen in Container` (with the '>'). This option will show up in the >< tab in the bottom left corner of vscode.

The above command will open a dev container using the docker image we provided.

There you go! 

### Docker Image Building Time
Building this docker image can take a while, for downloading MKL libraries and compiling SuiteSparse from the source code (just to remove a single print). 
In case you have a powerful workstation, considering changing all the `make -j8` to `make -j128`.

## Run Comparisons 

Comparisons with the Vector Heat Method can be found in the source files.

Results from Mancinelli & Puppo can be obtained using this modified code from
https://github.com/liyuesolo/RCM_on_meshes

## License
See the [LICENSE](./LICENSE) file for license rights and limitations.

## Contact
Please reach out to me ([Yue Li](yueli.cg@gmail.com)) if you spot any bugs or having quesitons or suggestions!

If this code contributes to your research, please consider citing our work:
```
@article{li2024differentiable,
  title={Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes},
  author={Li, Yue and Numerow, Logan and Thomaszewski, Bernhard and Coros, Stelian},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--14},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```