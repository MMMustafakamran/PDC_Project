::
::
::
:: This script requires two command line arguments: cxxparse.bat [Name of Source File] [Name of Dest. pdb File]
::
::

@ECHO OFF

IF NOT EXIST "%ROOT_PDT%\windows\bin\edgcpfe.exe" GOTO VARS_NOT_SET

IF "%1"=="" goto USAGE
IF "%2"=="" goto USAGE


IF NOT EXIST "%1" GOTO ARG_NOT_FOUND


%ROOT_PDT%\windows\bin\edgcpfe.exe -o %1.il "-I." "-I%VC_ROOT%\include" "-I%VC_ROOT%\PlatformSDK\include"  "-D__cdecl=" "-D__stdcall=" "-D__fastcall=" "-D__i386" "-D_MSC_VER=1310" "-D_W64="  "-D_INTEGRAL_MAX_BITS=64" "--exceptions" "--late_tiebreaker" "--no_class_name_injection" "--no_warnings" "--restrict" "--microsoft" "--new_for_init" %3 %4 %5 %6 %7 %8 %9 %1

IF NOT %ERRORLEVEL%==0 GOTO END


%ROOT_PDT%\windows\bin\taucpdisp.exe %1.il > %2


del %1.il > NUL
GOTO END

:USAGE
ECHO Usage: cxxparse [source-file] [output .pdb file]
ECHO (e.g. cxxparse test.cpp test.pdb)
GOTO END

:ARG_NOT_FOUND
ECHO could not find '%1'
GOTO END

:VARS_NOT_SET
ECHO Could not find edgcpfe.exe, did you edit and run pdtvars.bat?


:END
