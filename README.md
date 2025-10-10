# Framework di Reti Neurali Feed Forward

Questo progetto implementa un semplice framework in C per la creazione, l'addestramento e la valutazione di reti neurali feed-forward (MLP - Multi Layer Perceptron). Il codice è pensato per essere didattico e facilmente estendibile.

## Struttura del Progetto

- **nnet.h / nnet.c**: Definiscono le strutture dati principali (`NNet`, `NLayer`, `NNeuron`) e le funzioni per la gestione della rete neurale (inizializzazione, forward, backward, aggiornamento pesi, stampa).
- **funct.c**: Implementa le funzioni di utilità e le funzioni di attivazione (sigmoide, ReLU) e le loro derivate.
- **main.c**: Esempio di addestramento per la tabella AND.
- **or.c**: Esempio di addestramento per la tabella OR.
- **xor.c**: Esempio di addestramento per la tabella XOR.
- **Makefile**: Permette la compilazione semplice dei vari esempi.
- **README.md**: Questo file.

## Funzionalità Principali

- **Definizione della rete**: Puoi specificare la struttura della rete (numero di layer e neuroni per layer) tramite un array di interi.
- **Funzioni di attivazione**: Supporto per Sigmoide e ReLU, facilmente estendibili.
- **Addestramento**: Implementazione del backpropagation con discesa del gradiente.
- **Stampa**: Funzione per visualizzare la struttura e i parametri della rete.
- **Gestione dati**: Funzioni per creare e liberare array di dati di training.

## Esempi

Sono inclusi tre esempi di addestramento per le funzioni logiche AND, OR e XOR. Ogni esempio crea una rete, la addestra sui dati di verità e stampa i risultati.

### Esempio: XOR

Vedi [xor.c](xor.c):

```c
float TRAINING_DATA[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};
size_t init[] = {2, 3, 2, 1}; // 2 input, 2 hidden layer (3 e 2 neuroni), 1 output
NNet network = NetInit(init, size_of_array(init));
NetTrain(&network, data, TRAINING_COUNT, EPOCS, LRATE);
```

## Compilazione

Per compilare tutti gli esempi:

```sh
make
```

Per pulire i file generati:

```sh
make clean
```

## Esecuzione

Esegui uno degli esempi compilati:

```sh
./nnet         # AND
./nnet-or      # OR
./nnet-xor     # XOR
```

## Estensione

Puoi modificare la struttura della rete, la funzione di attivazione o i dati di training per sperimentare con altri problemi.

## Dipendenze

- GCC (compilatore C)
- Libreria matematica standard (`-lm`)

## Note

Il framework è pensato per scopi didattici e non è ottimizzato per grandi dataset o prestazioni elevate.

## Autore

Giacomo Mola 
