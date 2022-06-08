# Barney
## base
 * rouge e sas simili valori
 * context-label < context-chatbot: bot segue più il contesto rispetto al dialogo in se
 * perplexity alta approx 400 (rimandare)
 * emotion:
   * context-label elevata coincidenza, possibile spiegazione coerenza emotiva dei dialoghi?
   * ss ci mostra maggiore similitudine fra context e chatbot mentre emotion ci mostra maggiore similarità context e label. Possibili spiegazioni:
     * H1: in media i personaggi tendono ad avere lo stesso range emotivo 
     * H2: la metrica non è accurata
   * chatbot principalmente joy-anger-sadness
   * pearson
     * H1: metrica forse non riconosce i personaggi, tutto più dimile
     * H2: il fine tuning non accentua lo stile abbastanza
     * H3: l'emozione potrebbe non emergere chiaramente dai dati (script), ma dalla recitazione dei personaggi
 * bleu scarso
 * distinct non sembra avere grandi variazioni, molto più simile context-chatbot
 * ccl sempre fra 0.4 e 0.5
 * sc classifica bene i labels (con dev standard pressocché nulla testimone di un'alta convizione) mentre il chatbot si ferma a 0.4 (con dev standard altissima, alto grado di incertezza).

Da fare double check che rouge e sas siano simili, perché essendo algoritmica è interessante

## greedy_vs_nbeams
 * greedy e nbeams sono sono pressocché identici se non per bleu che è poco più basso

## greedy_vs_sampling
 * context greedy che è più alto, ci sta perché il greedy sia più vincolato al context. Il sampling più randomico ci sta. Anche l'emotion sono più o meno diversi
 * la forma della emotion è coerente con quanto visto in Barne.base perché era greedy.
 * da notare che è un po' più notevole la differenza nel sc (solo dato empirico, da confermere)

## nbeams_vs_sampling
 * sembra una copia di Barney.greedy_vs_sampling, unica differenza col sc ma la dev standard va da 0 a 1 quindi è sempre quasi casuale

N.B. la ccl è sempre fissa fra 0.4 e 0.5, ma quì si potrebbe anche togliere

## Barney vs Joey
 * La similitudine fra i due chatbot non è altissima, ma c'è 
 * Similarità Barney maggiore fra not-finetuned e Barney e minore Barney-Joy (solo dato empirico, da confermare)
 * il sc fallisce nel distinguere il fake da un altro personaggio perché i valori sono entrambi sotto 0.4 ma comunque la dev std è identica:
   * H1: potrebbe essere colpa del sc che non è forte 
   * H2: colpa del chatbot che non riesce a generare frasi simili al dataset

## Not finetuned
 * sc va meglio nel nf 

# Bender
## base
 * discorso simile a Barney.base su tutto, compresa la emotion (chatbot più sadness)
 * pearson poco più alta

N.B. La semantica non deve per forza essere correlata con le emozioni, è solo una un po' particolare ma non è contraddittoria

## greedy_vs_nbeams
 * stessi risultati per Barney.greedy_vs_nbeams
 * con differenza il sc che classifica leggermente meglio nbeams rispetto a greedy
N.B. controllare possibile problema di robustezza di sc perché le similarity sono a palletta mentre invece sc classifica 0.4 greedy e 0.6 nbemas (strano).

## greedy_vs_sampling
 * simile Barney.greedy_vs_sampling
 * sc sampling > greedy

## nbeams_vs_sampling
 * sc uguale

## Not finetuned
 * sc va meglio nel nf

---

# Fry
## base
 * discorso a Bender e Barney
 * il sc non ha dev std 0 su triplette(testsize of 250)
 * pearson correlation intorno a 0.5

## greedy_vs_nbeams
 * stessi risultati per Barney.greedy_vs_nbeams
 * con differenza il sc che classifica leggermente meglio nbeams rispetto a greedy
N.B. controllare possibile problema di robustezza di sc perché le similarity sono a palletta mentre invece sc classifica 0.4 greedy e 0.6 nbemas (strano).

## greedy_vs_sampling
 * stessi risultati, greedy più legato al context rispetto al sampling e un po' più sad
 * sc greedy > sampling sono tutti e due sopra 0.5

## nbeams_vs_sampling
 * nbeams più triste e più legato al context
 * sc nbeams < sampling, ma entrambi sopra 0.6
N.B. tra greedy_vs_sampling e nbeams_vs_sampling c'è una differenza significativa al sampling, dovuta molto probabilmente alla natura variabile del sampling

## Fry vs Bender
 * non sono così simili, ma il sc dice che Fry è meno Fry di Bender da notare però che vengono dallo stesso dataset
N.B. probabilmente le metriche non riescono

## Not finetuned
 * sc va meglio nel nf

---

# Harry

## base
 * sc sembra andare meglio del solito sul chatbot andando in positivo, mentre sul label sempre al 100%
 * perplexity esagerata
 * un po' più alta chatbot-label
 * label un po' più fear

N.B.: ccl sempre poco sotto a 0.4 o poco sopra a 0.5

## greedy_vs_nbeams
 * stessi risultati, sono identici
 * sc in generale abbastanza convinto che sia Harry in entrambi i casi

## greedy_vs_sampling
 * stessi risultati, greedy più legato al context rispetto al sampling, un po' più accentuato del solito
 * sc greedy < sampling sono tutti e due sopra 0.7

## nbeams_vs_sampling
 * copia di sampling

N.B. potrebbe esserci una relazione fra dimensione del dataset e sc -> sembra che sia migliore sui dataset piccoli (perché boh)

## Not finetuned
 * sc va meglio in Harry a differenza di tutti i casi sopra!

---

# Joey 
## base
 * molto gioiosa, ma chatbot è sempre un po' più triste
 * resto simile,
 * sc dev standard altissima

## greedy_vs_nbeams
 * sono identici, nbeams pelo meglio nel sc intorno a 0.6
 * bleu più basso rispetto al solito

## greedy_vs_sampling
 * stessi risultati, greedy più legato al context rispetto al sampling
 * emotion simili
 * sc greedy < sampling sono tutti e due sopra 0.6
N.B. Più o meno sempre soliti risultati a parte che per Fry

## nbeams_vs_sampling
 * sc nbeams > sampling sopra 0.5

N.B. potrebbe esserci una relazione fra dimensione del dataset e sc -> sembra che sia migliore sui dataset piccoli (perché boh)

## Not finetuned
 * sc va meglio Joey

---

# Phoebe
## base
 * Si vedono sempre le stesse cose aumenta la sadness nel chatbot e label-context stessa emotion

## greedy_vs_nbeams
 * molto meno anger e fear
 * nbeams > greedy

## greedy_vs_sampling
 * greedy più triste del sampling e più legato al contesto
 * sampling > greedy

## nbeams_vs_sampling
 * nbeams più triste e più legato al contesto
 * sc nbeams > sampling

## Not finetuned
 * Phoebe dipende più dal contesto
 * sc va meglio in Phoebe

---

# Sheldon
##
 * chatbot-label-context uguali -> metrica che fa schifo?
 * sc fake molto basso

## greedy_vs_nbeams
 * molto meno anger e fear
 * nbeams > greedy però sono bassissimi con valori sotto 0.2

## greedy_vs_sampling
 * greedy più legato al contesto
 * sampling > greedy

## nbeams_vs_sampling
 * nbeams più legato al contesto
 * sc nbeams < sampling

## Sheldon vs Barney
 * simile
 * sc Barney + Sheldoon di Sheldon, ma Sheldon di per sè ha un valore di sc bassissimo in generale
   * H: personaggio particolare per come parla e quindi non il chatbot non l'ha imparato

## Sheldon vs Fry
 * simile a Sheldon vs Barney anche per il sc

## Sheldon vs Phoebe
 * simile ma con la differenza che sc Sheldon è più Sheldon di Phoebe

## Not finetuned
 * sc va meglio Sheldon

---

# Vader
## base
 * non è triste ma anzi è gioioso O.o
 * sc eccellente su labels come sempre ma dev std altissima, il che dice che spara valori a caso 

## greedy_vs_nbeams
 * sono identici in tutto

## greedy_vs_sampling
 * greedy più legato al contesto
 * sampling > greedy

## nbeams_vs_sampling
 * nbeams più legato al contesto
 * sc sampling > nbeams
 * perfettamente identico a greedy_vs_sampling 

## Vader vs Bender
 * sc convinto che Vader sia Vader e che Bender non sia Vader

## Vader vs Harry
 * sc convinto che Vader sia Vader e che Harry non sia Vader

## Not finetuned	
 * sc va meglio in Vader!