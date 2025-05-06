
PDT 3.2 for Windows (Microsoft Visual Studio .NET 2003)

=============================================================================

To compile with the ductape library set your include directory to

pdtoolkit-3.2\ductape\inc

and add

pdtoolkit-3.2\Windows\lib

to your additional library directories.
Additionally, add "ductape.lib" to your list of dependencies, or
add "pragma comment (lib, "ductape.lib")" to your source file.

The sourcecode for the ductape library is provided in pdt\ductape\src

=============================================================================

You can remake the binaries using Microsoft Visual C++ .Net 2003 by executing
(with the proper variables from vcvars32.bat):

nmake /f Makefile.win32

=============================================================================

To use the provided binaries, edit pdtoolkit-3.2\Windows\bin\pdtvars.bat with your settings.
After setting up pdtvars.bat, simply run pdtvars.bat from a command window
and the binaries will be ready to go and in your path.

It is recommended that you add pdt\bin to your path.  Additionally, you
could set the varibles defined in pdtvars.bat in your startup.
