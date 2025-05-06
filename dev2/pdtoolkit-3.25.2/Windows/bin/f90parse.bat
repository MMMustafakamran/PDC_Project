::
::
::
:: This script requires two command line arguments: f90parse.bat [Name of Source File] [Name of Dest. pdb File]


@ECHO OFF

IF NOT EXIST "%ROOT_PDT%\bin\f90fe.exe" GOTO VARS_NOT_SET

IF "%1"=="" goto USAGE
IF "%2"=="" goto USAGE

IF NOT EXIST "%1" GOTO ARG_NOT_FOUND


%ROOT_PDT%\bin\f90fe.exe -o %1.il "-M . " %1

IF NOT %ERRORLEVEL%==0 GOTO END

%ROOT_PDT%\bin\pdtf90disp.exe %1.il > %2

del %1.il > NUL
GOTO END

:USAGE
ECHO Usage: f90parse [source-file] [output .pdb file]
ECHO (e.g. c90parse ring.f90 ring.pdb)
GOTO END

:ARG_NOT_FOUND
ECHO could not find '%1'
GOTO END

:VARS_NOT_SET
ECHO Could not find f90fe.exe, did you edit and run pdtvars.bat?


:END
