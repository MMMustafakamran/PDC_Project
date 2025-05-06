@ECHO OFF
::
:: Set ROOT_PDT to the location of pdt
:: Set VC_ROOT to the location of Visual C++
::
:: (e.g. 
::
::    set ROOT_PDT=E:\pdt
::    set VC_ROOT=E:\Program Files\Microsoft Visual Studio .NET 2003\Vc7
::  )
::


set ROOT_PDT=C:\pdtoolkit
set VC_ROOT=C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7


:: DO NOT EDIT BELOW THIS LINE
set PATH=%PATH%;%ROOT_PDT%\windows\bin
ECHO PDT Variables are now set
ECHO ROOT_PDT=%ROOT_PDT%
ECHO VC_ROOT=%VC_ROOT%
