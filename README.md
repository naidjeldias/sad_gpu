# sad_gpu

Algoritmo para criação de mapas de disparidade tanto na CPU quanto na GPU utilizando CUDA.

### Dependências
* Opencv versão 4.x.x (não foi testado em versões inferiores mas talvez possa funcionar)
* CUDA 10 ou 11 (pode funcionar em outras versões do CUDA)
* CMake versão 8 ou inferior ( versões acima não conseguem compilar o programa)


### Compilação
```
mkdir -p build/
cd build/
cmake -D WITH_DISPLAY=OFF ..
make 

```
Ou pode simplesmente utilizar o script `build.sh`

### Execução
` ./build/disparity <TAMANHO_JANELA> <DISPARIDADE_MAXIMA> [<RESOLUCAO> (F - full size; H - Half size)]`

Exemplo: `./build/disparity 5 64`