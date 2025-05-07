# Setup
**NOTE** There is currently a GCC bug that prevents further performance increases
as prefetching causes vectorization to fail. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120162.

Clone the github repository using `git clone --recursive`, to make sure you get all the submodules.

If you are on Linux/Mac:
* Depending on your architecture, run either `bootstrap_linux_mac_x86.sh` or `bootstrap_linux_mac_ARM.sh` (making sure you can run it using `chmod +x bootstrap_linux_mac_*.sh`)
* In your build folder you can now run `make` in the future if you want to recompile after making changes to the code.
* Execute `./GAME` to run the project (making sure you can run it using `chmod +x GAME`)

If you are on windows:
**NOTE** this has not been tested with MSVC, compiler options are **NOT** optimal
and performance **WILL** be worse. *Please only compile this with GCC.*

**CMake has been set to generate an error on windows.** This error message can be removed
and it should compile fine.

* Run the `bootstrap_windows.bat` file (this requires Visual Studio 17 2022 by default).
* Open the generated Visual Studio project located in the build folder.
* Run the project in Release mode, making sure you select GAME as the build target.


### Notes
Has been tested on:
* (Manjaro) Linux x86
