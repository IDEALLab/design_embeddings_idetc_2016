@echo off

if exist .\temp (goto ask) else (
    mkdir .\temp
    goto start_mongodb
)

:ask
echo Do you want to cleanup temp files?(Y/N)
set INPUT=
set /P INPUT=%=%
if /I "%INPUT%"=="y" goto yes
if /I "%INPUT%"=="yes" goto yes
if /I "%INPUT%"=="n" goto no
if /I "%INPUT%"=="no" goto no
goto ask

:yes
del /q /s temp\*
echo finished!
goto start_mongodb

:no
echo continue...
goto start_mongodb

:start_mongodb
echo mongodb running...
mongod --logpath .\temp\mongodb.log --dbpath .\temp