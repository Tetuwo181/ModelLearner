@echo off

setlocal
set url=https://colab.research.google.com/drive/1ACnzRp_aU_6Y9Ul3O3HxBUNfhmDarAnZ#scrollTo=gWsG_o-0zQJG

for /l %%i in (1,1,12) do (
  timeout 1200 /nobreak && echo %%i && start %url%
)
endlocal