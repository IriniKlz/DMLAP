## **Python Setup**

When it comes to installing Python, there can be many ways of implementing that. The install instructions below, follow the paradigm of [PyFAD](https://github.com/colormotor/PyFAD/tree/main) and [PoMa](https://github.com/jchwenger/poetic.machines/tree/main) so that students who attended those modules in term I, continue with the same setup.

### **1. Installing Python with Miniforge**

We will use [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to install Python efficiently. 

Miniforge is a lightweight distribution of [Conda](https://docs.conda.io/projects/conda/en/latest/), designed to simplify package management and environment handling for Python users. Unlike the standard [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Anaconda](https://www.anaconda.com/) distributions, Miniforge is built around the Conda-Forge community-maintained package repository, which provides up-to-date and optimised packages. It also includes [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), a faster alternative to Conda for package installation.

Once Miniforge is installed, you will be able to use `conda / mamba` in a terminal and move into **Creating your Environment** (see below).

This approach works across Windows, macOS, and Linux. On Windows, I recommend allowing Miniforge to be your primary Python provider and adding it to the PATH of all terminals (so you can access `conda / mamba` in them).

### Checking your Installation

For Python, open a terminal and type `python --version` and/or `python3 --version`.

Note: to locate programs, and see if they are installed, use `which` on Linux/MacOs, `where` on Windows.

In a terminal, type `where/which conda`.

If it says something to the effect of `conda not found`, then you are good to continue with your installation

Note: if you install things, often you need to reset or close the current terminal, then reopen a new one.

### **2. Creating your DMLAP Environment**

Setting up a Python environment (think of it as an ecosystem of packages) can sometimes feel like navigating a maze, as illustrated in the comic below. To simplify this process, we use `conda / mamba` that allows for easy installation, updating, and dependency management while enabling seamless switching between different Python environments on your local computer.

![a fun representation of the maze of setting up a virtual environment](https://imgs.xkcd.com/comics/python_environment.png)

Now, open the terminal and install the module's dependencies:

```bash
mamba env create -f environment.yaml
```
When asked `[Y/n]` type `y` and enter. It is a confirmation to proceed with the installation.

If you need to update the environment using the YAML file, do this:

```bash
mamba env update -f environment.yaml
```
Note that ``conda`` and ``mamba`` can be used interchangeably, but `mamba` allows for faster execution.

You just created an environment called `dmlap`! This is the environment you will have to activate (see instructions below) and use for any process related to this module. 

### **3. Working with Conda Environments**

When using `conda`, it is **strongly recommended never to use the `base` environment**, the one coming out of the box with `conda`. Just leave it as is and always work in an environment you created (like `dmlap`).

If you don't want to have the `base` environment automatically activated, run this:

```bash
$ mamba config --set auto_activate_base false
```

The following instructions will get you through the ways of using `conda / mamba` to work with your environments. You can list all `conda / mamba` flags with:

```bash
$ mamba --help`
``` 

### Creating an Environment

Apart from creating an environment through a YAML file, as we did for creating `dmlap`, you can also do the following.

Open a terminal/console and type: 

```bash
# using mamba is faster than conda
$ mamba create --name test-env python
```
`--name` and `-n` are equivalent

This will specify the Python version and its name 'test-env'.

Activate it by typing:

```bash
$ mamba activate test-env
```

Now your console *should* indicate the environment in some way or other. We can install a package (the `-c conda-forge` specifies a *channel*, again a way of controlling dependencies: a bit like getting your version of the program from one supermarket). You can add the `-y` flag if you don't want conda to wait for your approval.

```bash
(test-env) $ mamba install -c conda-forge scipy
```

The package is now installed **only** in your environment `test-env`. 

### Checking what Packages are Installed in your Environment

```bash
(test-env)  $ mamba list
```

Search for something in particular, using Unix pipes:

```bash
(test-env) $ mamba list | grep scipy # pytorch, torchvision, etc.
```

### Removing Packages from your Environment

Remove the package `scipy` from the currently active environment:

```bash
(test-env) $ mamba remove scipy
```

Remove a list of packages:

```bash
(test-env) $ mamba remove -n test-env scipy curl wheel
```

Remove all packages:

```bash
$ mamba remove -n test-env --all
```

Remove the environment altogether: 

```bash
$ mamba env remove -n test-env
```

### Listing all your Environments

```bash
$ mamba env list
```

### More Information:

- [Getting Started with Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

### **4. Using Python in VSCode**

1. Install VSCode and extensions: **Python**, **Jupyter** and **Markdown All in One**

2. Open a Jupyter Notebook (`.ipynb`)

3. On the top right corner of your window, click **Select Kernel** > **Select Environment** and choose your Miniforge environment

4. Open a code cell and run:
   ```python
   print("Hello world")
   ```

### **5. Troubleshooting**

If something goes wrong, delete the Miniforge directory (a local folder for your current user) and reinstall.

