# nnet — Framework minimale per reti neurali feed‑forward (MLP)

Descrizione
-----------
Piccolo framework in C per sperimentare reti neurali feed‑forward (MLP). È pensato per scopi didattici: definizione della topologia, forward, backpropagation, aggiornamento pesi (SGD e Adam), inizializzazione personalizzabile (es. Xavier) e gestione della memoria.

Requisiti
---------
- GCC o altro compilatore C compatibile C99
- Libreria matematica standard (-lm)
- Sistema Linux (testato)

Struttura del repository
------------------------
- `nnet.h`, `nnet.c` — strutture dati e implementazione core (NetInit, NetEvaluate, NetBack, NetUpdate, NetUpdateAdam, NetTrain, NetFree, ecc.)
- `funct.c` — funzioni di attivazione e derivate (SIGMOID, RELU, LRELU, TANH)
- `xor.c`, `and.c`, `or.c` — esempi di utilizzo / test
- `Makefile` — target di compilazione

Caratteristiche principali
--------------------------
- Definizione dinamica della topologia tramite array di size_t (es. {2,3,1})
- Funzioni di attivazione: SIGMOID, TANH, RELU, LRELU
- Inizializzazione pesi personalizzabile (possibilità di passare una funzione, es. Xavier)
- Ottimizzatore Adam implementato con bias‑correction (NetUpdateAdam)
- Aggiornamento pesi via batch (accumulo gradiente)—estendibile a mini‑batch
- NetFree per il rilascio completo della memoria allocata

API essenziale
--------------
- NNet NetInit(size_t *netsize, size_t size, float(*randfloat)())
  - Crea la rete; `randfloat()` deve restituire un float nell'intervallo desiderato.
- NNet* NetEvaluate(NNet *nn, float *input)
  - Esegue forward e aggiorna le attivazioni interne.
- NNet* NetBack(NNet *nn, float *output)
  - Calcola i gradienti (backpropagation) rispetto all'output target.
- NNet* NetUpdate(NNet *nn, int rows, float lr)
  - Aggiorna pesi e bias con SGD (batch).
- NNet* NetUpdateAdam(NNet *nn, int rows, float lr, float beta1, float beta2, float eps)
  - Aggiorna pesi e bias usando Adam con correzione dei bias.
- float NetCost(NNet *nn, float **data, int rows)
  - Calcola la loss media (MSE).
- NNet* NetTrain(NNet *nn, float **data, int rows, int epocs, float lr)
  - Loop di training che usa l'optimizer impostato nella rete.
- void NetFree(NNet *nn)
  - Libera tutte le strutture interne allocate.
- Helper: NetMakeDataArray / NetFreeDataArray

Inizializzazione Xavier
-----------------------
Per migliorare la convergenza è disponibile un helper di inizializzazione Xavier. La funzione passata a `NetInit` dovrebbe restituire valori casuali opportunamente scalati (es. ~ sqrt(6/(fan_in+fan_out))). Passare `&xavier_rand` a `NetInit` per usarla.

Ottimizzatore Adam (breve)
--------------------------
NetUpdateAdam applica le seguenti operazioni per ogni parametro w:
- m_t = beta1*m_{t-1} + (1-beta1)*g
- v_t = beta2*v_{t-1} + (1-beta2)*g^2
- m̂ = m_t / (1 - beta1^t)
- v̂ = v_t / (1 - beta2^t)
- w = w - lr * m̂ / (sqrt(v̂) + eps)

Parametri raccomandati: beta1=0.9, beta2=0.999, eps=1e-8. Net mantiene il contatore t per la correzione dei bias.

Esempio consigliato: XOR
------------------------
Suggerimenti per far convergere rapidamente XOR:
- Topologia: 2-2-1 o 2-3-1
- Inizializzazione: Xavier
- Hidden: TANH o LRELU
- Output: SIGMOID (target 0/1)
- Optimizer: Adam
- Learning rate consigliato: 0.001 - 0.01 (Adam), 0.01 - 0.05 (SGD)
- Se la loss si ferma a 0.25 significa che la rete prevede ≈0.5 per tutti i sample (simmetria o pesi bloccati).

Esempio minimale (uso)
```c
size_t init[] = {2, 2, 1};
NNet net = NetInit(init, 3, &xavier_rand);

/* set activation */
net.layers[1].funct = &TANH;         /* hidden */
net.layers[net.size-1].funct = &SIGMOID; /* output */

net.optimizer = OPTIMIZER_ADAM;
NetTrain(&net, data, TRAINING_COUNT, 100000, 0.01f);
NetFree(&net);
```
