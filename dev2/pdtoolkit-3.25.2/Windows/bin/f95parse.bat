::
::
::
:: This script requires two command line arguments: f95parse.bat [Name of Source File] [Name of Dest. pdb File]
::
::

@ECHO OFF

IF NOT EXIST "%ROOT_PDT%\bin\pdtflint.exe" GOTO VARS_NOT_SET

IF "%1"=="" goto USAGE
IF "%2"=="" goto USAGE


IF NOT EXIST "%1" GOTO ARG_NOT_FOUND

SET FLINTHOME=%ROOT_PDT%\..\etc
%ROOT_PDT%\bin\pdtflint.exe %1 -Mpdb=%2 %3 %4 %5 %6 %7 %8 %9

GOTO END

:USAGE
ECHO Usage: f95parse.bat [source-file] [output .pdb file]
ECHO (e.g. f95parse ring.f90 ring.pdb)
GOTO END

:ARG_NOT_FOUND
ECHO could not find '%1'
GOTO END

:VARS_NOT_SET
ECHO Could not find pdtflint.exe, did you edit and run pdtvars.bat?


:END
