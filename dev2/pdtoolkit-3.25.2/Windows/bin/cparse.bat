::
::
::
:: This script requires two command line arguments: cparse.bat [Name of Source File] [Name of Dest. pdb File]
::
::

@ECHO OFF

IF NOT EXIST "%ROOT_PDT%\bin\edgcpfe.exe" GOTO VARS_NOT_SET

IF "%1"=="" goto USAGE
IF "%2"=="" goto USAGE


IF NOT EXIST "%1" GOTO ARG_NOT_FOUND

%ROOT_PDT%\bin\edgcpfe.exe -o %1.il "-I%VC_ROOT%\include" "-I%VC_ROOT%" "-I%VC_ROOT%\PlatformSDK\include" "-D__cdecl=" "-D__stdcall=" "-D__fastcall=" "-D__i386" "-D_MSC_VER=1310" "-D_W64=" "-D_INTEGRAL_MAX_BITS=64" "--no_warnings" "--c99" %3 %4 %5 %6 %7 %8 %9 %1

IF NOT %ERRORLEVEL%==0 GOTO END


%ROOT_PDT%\bin\taucpdisp.exe %1.il > %2


del %1.il > NUL
GOTO END

:USAGE
ECHO Usage: cparse [source-file] [output .pdb file]
ECHO (e.g. cparse test.cpp test.pdb)
GOTO END

:ARG_NOT_FOUND
ECHO could not find '%1'
GOTO END

:VARS_NOT_SET
ECHO Could not find edgcpfe.exe, did you edit and run pdtvars.bat?


:END



