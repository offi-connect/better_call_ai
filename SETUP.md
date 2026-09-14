\# better\_call\_ai



\## Procedura per ricreare, avviare e sincronizzare il progetto su un altro PC



Repository GitHub:



`https://github.com/offi-connect/better\_call\_ai.git`



La procedura seguente permette di ricreare il progetto su un nuovo PC, installare l'ambiente Python, avviare l'applicazione e successivamente salvare le modifiche su GitHub.



\---



\## 1. Prerequisiti



Sul nuovo PC devono essere installati:



\* \*\*Git\*\*

\* \*\*Python 3.x\*\*



Non è necessario copiare il progetto dal vecchio PC: viene recuperato da GitHub.



\---



\# PARTE A — Prima installazione sul nuovo PC



\## 2. Aprire PowerShell



Aprire PowerShell e spostarsi nella cartella in cui si vuole installare il progetto.



Esempio:



```powershell

cd C:\\Dev

```



\---



\## 3. Clonare il repository



Eseguire:



```powershell

git clone https://github.com/offi-connect/better\_call\_ai.git

```



Entrare nella cartella:



```powershell

cd C:\\Dev\\better\_call\_ai

```



\---



\## 4. Verificare il collegamento con GitHub



Eseguire:



```powershell

git status

```



Il risultato dovrebbe essere simile a:



```text

On branch main

Your branch is up to date with 'origin/main'.



nothing to commit, working tree clean

```



Se compare questo messaggio, il progetto locale è correttamente collegato a GitHub.



\---



\# PARTE B — Creare l'ambiente Python



\## 5. Creare il virtual environment



Dalla cartella:



```text

C:\\Dev\\better\_call\_ai

```



eseguire:



```powershell

python -m venv .venv

```



Questo crea:



```text

better\_call\_ai

└── .venv

```



La cartella `.venv` NON deve essere caricata su GitHub.



\---



\## 6. Attivazione del virtual environment



Normalmente:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Se PowerShell restituisce:



```text

L'esecuzione di script è disabilitata nel sistema in uso

```



non è un problema del progetto.



È possibile utilizzare direttamente il Python contenuto nel virtual environment senza attivarlo.



In questo caso utilizzare sempre:



```powershell

.\\.venv\\Scripts\\python.exe

```



Per esempio:



```powershell

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



Questa modalità è quella utilizzata nella procedura di test del progetto.



\---



\# PARTE C — Installare le dipendenze



\## 7. Installare requirements.txt



Eseguire:



```powershell

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



Il comando installa automaticamente le librerie necessarie, tra cui:



\* `sounddevice`

\* `numpy`

\* `faster-whisper`

\* `requests`

\* `anthropic`



Non è necessario installarle singolarmente.



\---



\# PARTE D — Avviare l'applicazione



\## 8. Avviare better\_call\_ai



Dalla cartella del progetto:



```powershell

.\\.venv\\Scripts\\python.exe verbale.py

```



L'applicazione dovrebbe aprire la propria finestra.



Il terminale può non mostrare messaggi: è normale.



\---



\## 9. Primo avvio di faster-whisper



Al primo utilizzo, `faster-whisper` potrebbe scaricare automaticamente il modello configurato.



Il modello utilizzato dipende dal file:



```text

config.json

```



Nel progetto attuale il modello locale configurato è:



```text

medium

```



Il download può richiedere tempo e spazio su disco.



Dopo il primo download, gli avvii successivi saranno più veloci.



\---



\# PARTE E — Verificare che la registrazione funzioni



\## 10. Fare una registrazione di prova



Nella finestra dell'applicazione:



1\. premere \*\*Avvia\*\*;

2\. parlare per 10–15 secondi;

3\. premere \*\*Stop\*\*.



Il programma dovrebbe creare una nuova cartella dentro:



```text

sessioni\\

```



con una struttura simile a:



```text

sessioni\\

└── 2026-09-14\_18-30-00\\

&#x20;   ├── verbale.md

&#x20;   └── audio.wav

```



`verbale.md` contiene la trascrizione.



`audio.wav` contiene la registrazione audio.



Se questi file vengono creati correttamente, l'installazione è funzionante.



\---



\# PARTE F — API KEY



Le API key NON devono essere salvate nel repository GitHub.



Il progetto utilizza variabili d'ambiente.



Per OpenAI:



```powershell

$env:OPENAI\_API\_KEY="LA\_TUA\_API\_KEY"

```



Per Anthropic:



```powershell

$env:ANTHROPIC\_API\_KEY="LA\_TUA\_API\_KEY"

```



Le API key vanno configurate sul singolo PC.



Non inserire mai direttamente la chiave in:



```text

config.json

```



o nel codice Python.



\---



\# PARTE G — Prima modifica e commit



Una volta modificato il progetto, verificare cosa è cambiato:



```powershell

git status

```



Per vedere le differenze nel codice:



```powershell

git diff

```



\---



\## 11. Aggiungere le modifiche



Quando si è sicuri delle modifiche:



```powershell

git add .

```



Controllare cosa verrà committato:



```powershell

git status

```



\---



\## 12. Creare il commit



Esempio:



```powershell

git commit -m "Modifica gestione trascrizione"

```



Il messaggio deve descrivere brevemente cosa è stato modificato.



Esempi:



```powershell

git commit -m "Corregge segmentazione audio"

```



```powershell

git commit -m "Migliora gestione verbale"

```



```powershell

git commit -m "Aggiorna configurazione trascrizione"

```



\---



\## 13. Inviare il commit su GitHub



Eseguire:



```powershell

git push

```



Non è necessario specificare `origin main` ogni volta perché il branch locale `main` è già collegato a:



```text

origin/main

```



\---



\# PARTE H — Tornare a lavorare sul progetto da un altro PC



Se il progetto è già stato clonato sul PC:



\## Prima di iniziare a lavorare



Eseguire:



```powershell

git pull

```



Questo scarica eventuali modifiche fatte da altri PC.



Poi avviare:



```powershell

.\\.venv\\Scripts\\python.exe verbale.py

```



\---



\## Dopo aver lavorato



Eseguire:



```powershell

git status

```



poi:



```powershell

git add .

```



poi:



```powershell

git commit -m "Descrizione della modifica"

```



e infine:



```powershell

git push

```



\---



\# WORKFLOW DA RICORDARE



\## Nuovo PC — prima volta



```powershell

cd C:\\Dev

git clone https://github.com/offi-connect/better\_call\_ai.git

cd better\_call\_ai

python -m venv .venv

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

.\\.venv\\Scripts\\python.exe verbale.py

```



\## PC già configurato — inizio lavoro



```powershell

cd C:\\Dev\\better\_call\_ai

git pull

.\\.venv\\Scripts\\python.exe verbale.py

```



\## Dopo aver modificato il progetto



```powershell

git status

git add .

git commit -m "Descrizione della modifica"

git push

```



\---



\# IMPORTANTE — File audio delle sessioni



Le registrazioni generate dall'applicazione vengono salvate in:



```text

sessioni\\

```



I file `audio.wav` possono diventare molto grandi.



Il repository originale contiene alcuni file audio, ma il file da circa 140 MB della sessione:



```text

sessioni/2026-07-26\_10-44-33/audio.wav

```



è stato deliberatamente escluso dal repository perché supera il limite di dimensione di GitHub.



Pertanto:



\*\*GitHub contiene il codice del progetto, ma non necessariamente tutti gli audio delle sessioni.\*\*



Gli audio che devono essere conservati non dovrebbero essere caricati indiscriminatamente su GitHub.



\---



\# REGOLA FONDAMENTALE



Prima di fare:



```powershell

git add .

```



controllare sempre:



```powershell

git status

```



In particolare verificare che non vengano aggiunti accidentalmente:



\* API key

\* file temporanei

\* `.venv`

\* file audio molto grandi

\* cache Python

\* dati personali o sensibili



\---



\# Stato finale atteso



Alla fine della procedura il PC dovrebbe avere:



```text

C:\\Dev\\better\_call\_ai\\

│

├── .git\\

├── .venv\\

├── sessioni\\

├── .gitignore

├── README.md

├── requirements.txt

├── config.json

├── verbale.py

├── bench.py

├── test\_segmenter.py

├── test\_summarizer.py

└── altri file del progetto

```



Il progetto locale sarà collegato a:



```text

https://github.com/offi-connect/better\_call\_ai.git

```



e il lavoro potrà essere sincronizzato tra PC tramite:



```text

git pull

```



per ricevere modifiche e:



```text

git push

```



per pubblicare le proprie modifiche.



