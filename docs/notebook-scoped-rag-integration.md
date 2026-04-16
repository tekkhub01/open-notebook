# Integrazione RAG per Notebook Specifici

**Data:** 2026-02-12  
**Versione:** 1.0  
**Autore:** Analisi tecnica su Open Notebook

---

## 📋 Problema

Attualmente il sistema RAG di Open Notebook effettua ricerche su **tutte le fonti** nel database senza possibilità di filtrare per notebook specifico.

**Limitazione attuale:**
- Un cliente necessita di un RAG specifico → bisogna clonare l'intero progetto
- Impossibile avere più clienti/progetti con RAG isolati nello stesso database

**Obiettivo:**
Implementare la possibilità di effettuare ricerche RAG limitate a un singolo notebook, mantenendo la compatibilità con ricerche globali.

---

## 🔍 Analisi Struttura Esistente

### Database SurrealDB

**Namespace:** `open_notebook`  
**Database:** `open_notebook`

#### Tabelle Principali

```
source                 → Documenti caricati
source_embedding       → Chunk embeddings (vettori)
notebook              → Notebook/progetti
reference             → Relazione IN source OUT notebook (GRAPH EDGE)
```

#### Funzioni di Ricerca Attuali

1. **`fn::text_search()`** - Full-text search (BM25)
   - Parametri: `$query_text, $match_count, $sources, $show_notes`
   - Cerca in: `source.title`, `source.full_text`, `source_embedding.content`, `source_insight.content`

2. **`fn::vector_search()`** - Vector similarity search
   - Parametri: `$query, $match_count, $sources, $show_notes, $min_similarity`
   - Cerca in: `source_embedding.embedding`, `source_insight.embedding`, `note.embedding`

**Problema:** Nessun filtro per notebook!

---

## ✅ Situazione Corrente (verificata)

### Dati Esistenti

```sql
-- 9 source totali
SELECT count() FROM source GROUP ALL;

-- 9 reference (tutte collegate)
SELECT count() FROM reference GROUP ALL;

-- 1 notebook attivo
SELECT id, title FROM notebook;
-- Risultato: notebook:wcey1gczvhr6vbdxyyn5
```

### Reference Popolate

Tutte le 9 source sono già collegate al notebook principale tramite la tabella `reference`:

```sql
reference:1x19dridkymklntghlan → source:tnn1sfdhe5y5ck30611c → notebook:wcey1gczvhr6vbdxyyn5
reference:4kzjmy9ujnz8a8jrm7le → source:o96lyfb7pgus91ukip88 → notebook:wcey1gczvhr6vbdxyyn5
...
```

**✅ Base dati pronta per l'integrazione!**

---

## 🛠️ Piano di Implementazione

### Step 1: Modifica Funzioni SurrealDB

#### 1.1 Modifica `fn::text_search()`

**Aggiungere parametro:**
```sql
$notebook_id: option<record<notebook>>
```

**Modificare query per ogni tabella:**

```sql
-- Source title search
LET $source_title_search = IF $sources { 
  (SELECT id, title, search::highlight('`', '`', 1) AS content, 
          id AS parent_id, math::max(search::score(1)) AS relevance 
   FROM source 
   WHERE title @1@ $query_text
   AND ($notebook_id = NONE OR id IN (SELECT in FROM reference WHERE out = $notebook_id))
   GROUP BY id) 
} ELSE { [] };

-- Source embedding search
LET $source_embedding_search = IF $sources { 
  (SELECT source.id AS id, source.title AS title, 
          search::highlight('`', '`', 1) AS content, 
          source.id AS parent_id, math::max(search::score(1)) AS relevance 
   FROM source_embedding 
   WHERE content @1@ $query_text
   AND ($notebook_id = NONE OR source IN (SELECT in FROM reference WHERE out = $notebook_id))
   GROUP BY id) 
} ELSE { [] };

-- Ripetere per: source_full_search, source_insight_search
```

**Note search:** mantenere invariato (notes non hanno reference)

#### 1.2 Modifica `fn::vector_search()`

**Aggiungere parametro:**
```sql
$notebook_id: option<record<notebook>>
```

**Modificare query:**

```sql
-- Source embedding search
LET $source_embedding_search = IF $sources { 
  (SELECT source.id AS id, source.title AS title, content, 
          source.id AS parent_id, 
          vector::similarity::cosine(embedding, $query) AS similarity 
   FROM source_embedding 
   WHERE embedding != NONE 
   AND array::len(embedding) = array::len($query) 
   AND vector::similarity::cosine(embedding, $query) >= $min_similarity
   AND ($notebook_id = NONE OR source IN (SELECT in FROM reference WHERE out = $notebook_id))
   ORDER BY similarity DESC
   LIMIT $match_count) 
} ELSE { [] };

-- Ripetere per: source_insight_search
```

---

### Step 2: Modifica Backend API

#### Endpoint: `POST /api/search/ask`

**Request Schema Attuale:**
```json
{
  "query": "string",
  "match_count": 5,
  "sources": true,
  "show_notes": true
}
```

**Request Schema Nuovo:**
```json
{
  "query": "string",
  "match_count": 5,
  "sources": true,
  "show_notes": true,
  "notebook_id": "notebook:wcey1gczvhr6vbdxyyn5"  // ← NUOVO (opzionale)
}
```

**Logica Backend:**
```python
# Pseudo-codice
def ask_knowledge_base(request):
    query = request.query
    notebook_id = request.notebook_id or None  # None = ricerca globale
    
    # Chiamata alle funzioni SurrealDB
    text_results = db.query(
        "fn::text_search",
        query_text=query,
        match_count=request.match_count,
        sources=request.sources,
        show_notes=request.show_notes,
        notebook_id=notebook_id  # ← NUOVO PARAMETRO
    )
    
    # ... stesso per vector_search
```

**Backward Compatibility:**
- Se `notebook_id` non viene passato → `None` → ricerca globale (come prima)
- Mantenere tutti i test esistenti funzionanti

---

### Step 3: Modifica Frontend

#### UI Necessaria

**Componente Selector Notebook:**
```javascript
// Fetch notebooks disponibili
const notebooks = await fetch('/api/notebooks').then(r => r.json());

// Dropdown/Select
<select id="notebookSelector">
  <option value="">Tutti i notebook</option>
  {notebooks.map(nb => (
    <option value={nb.id}>{nb.title || nb.id}</option>
  ))}
</select>
```

**Request con notebook_id:**
```javascript
const selectedNotebookId = document.getElementById('notebookSelector').value;

const response = await fetch('/api/search/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: userQuery,
    match_count: 10,
    sources: true,
    show_notes: true,
    notebook_id: selectedNotebookId || null  // null = tutti
  })
});
```

**Persistenza scelta:**
```javascript
// Salvare in localStorage
localStorage.setItem('selectedNotebook', selectedNotebookId);

// Ripristinare all'avvio
const savedNotebook = localStorage.getItem('selectedNotebook');
```

---

## 📝 Checklist Implementazione

### Database
- [ ] Modificare `fn::text_search()` con parametro `$notebook_id`
- [ ] Aggiungere filtro notebook a tutte le query `source*`
- [ ] Modificare `fn::vector_search()` con parametro `$notebook_id`
- [ ] Testare funzioni con `notebook_id = NONE` (ricerca globale)
- [ ] Testare funzioni con `notebook_id` specifico

### Backend
- [ ] Aggiornare schema request `/api/search/ask`
- [ ] Passare `notebook_id` alle funzioni SurrealDB
- [ ] Aggiornare documentazione OpenAPI/Swagger
- [ ] Creare endpoint `GET /api/notebooks` (se non esiste)
- [ ] Test backward compatibility (request senza `notebook_id`)
- [ ] Test con `notebook_id` valido
- [ ] Test con `notebook_id` inesistente (error handling)

### Frontend
- [ ] Implementare selector notebook
- [ ] Gestire stato "Tutti i notebook" (null)
- [ ] Salvare scelta in localStorage
- [ ] Aggiungere feedback visivo (quale notebook è selezionato)
- [ ] Gestire errori (notebook non trovato)

---

## 🧪 Test Cases

### Test 1: Ricerca Globale (Backward Compatibility)
```json
POST /api/search/ask
{
  "query": "dirigente responsabile",
  "match_count": 5,
  "sources": true,
  "show_notes": true
  // notebook_id omesso
}
```
**Risultato atteso:** Tutte le 9 source vengono considerate (come prima)

---

### Test 2: Ricerca su Notebook Specifico
```json
POST /api/search/ask
{
  "query": "dirigente responsabile",
  "match_count": 5,
  "sources": true,
  "show_notes": true,
  "notebook_id": "notebook:wcey1gczvhr6vbdxyyn5"
}
```
**Risultato atteso:** Solo source collegate a quel notebook

---

### Test 3: Notebook Vuoto
```sql
-- Creare notebook senza source
CREATE notebook:test_empty SET title = "Test Empty";
```
```json
POST /api/search/ask
{
  "query": "qualsiasi cosa",
  "notebook_id": "notebook:test_empty"
}
```
**Risultato atteso:** Array vuoto (nessun risultato)

---

### Test 4: Notebook Inesistente
```json
POST /api/search/ask
{
  "query": "test",
  "notebook_id": "notebook:nonexistent"
}
```
**Risultato atteso:** 404 o array vuoto (gestire errore)

---

## 🎯 Scenario Multi-Cliente

### Setup Multi-Tenant

**Cliente A:**
```sql
-- Creare notebook
CREATE notebook:clienteA SET title = "Cliente A - Regolamenti Calcio";

-- Collegare source
INSERT INTO reference (in, out) VALUES 
  (source:doc1, notebook:clienteA),
  (source:doc2, notebook:clienteA);
```

**Cliente B:**
```sql
CREATE notebook:clienteB SET title = "Cliente B - Pallavolo";

INSERT INTO reference (in, out) VALUES 
  (source:doc3, notebook:clienteB),
  (source:doc4, notebook:clienteB);
```

**Source Condivise:**
```sql
-- Una source in più notebook (es. regole generali CSI)
INSERT INTO reference (in, out) VALUES 
  (source:norme_generali, notebook:clienteA),
  (source:norme_generali, notebook:clienteB);
```

### Query Isolation

```javascript
// Cliente A - vede solo doc1, doc2, norme_generali
fetch('/api/search/ask', {
  body: JSON.stringify({
    query: "arbitro",
    notebook_id: "notebook:clienteA"
  })
});

// Cliente B - vede solo doc3, doc4, norme_generali  
fetch('/api/search/ask', {
  body: JSON.stringify({
    query: "arbitro",
    notebook_id: "notebook:clienteB"
  })
});
```

---

## 📊 Stima Impatto

### Difficoltà
⭐⭐☆☆☆ (2/5 - Media-Bassa)

### Tempo Stimato
- **Database (SQL):** 1-2 ore
- **Backend (Python):** 1-2 ore  
- **Frontend:** 1-2 ore
- **Testing:** 2-3 ore

**Totale:** ~6-9 ore per un dev che conosce il progetto

### Breaking Changes
❌ Nessuno (backward compatible)

### Performance Impact
✅ Minimo (aggiunge un filtro su tabella indicizzata)

---

## 🚀 Deployment

### 1. Deploy Database
```sql
-- Backup prima di modificare
surreal export --conn http://localhost:8000 \
  --user root --pass root \
  --ns open_notebook --db open_notebook \
  backup_pre_notebook_filter.sql

-- Applicare modifiche alle funzioni
-- (eseguire script SQL con le nuove definizioni)
```

### 2. Deploy Backend
```bash
# Aggiornare dipendenze se necessario
pip install -r requirements.txt

# Restart service
systemctl restart open-notebook-backend
```

### 3. Deploy Frontend
```bash
# Build e deploy del frontend custom
npm run build
# ... deploy su server
```

### 4. Verifica
```bash
# Test ricerca globale
curl -X POST http://localhost:5055/api/search/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "match_count": 5}'

# Test ricerca filtrata
curl -X POST http://localhost:5055/api/search/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "notebook_id": "notebook:wcey1gczvhr6vbdxyyn5"}'
```

---

## 📚 Risorse

### Documentazione SurrealDB
- [Graph Relations](https://surrealdb.com/docs/surrealql/statements/relate)
- [Functions](https://surrealdb.com/docs/surrealql/functions)
- [Indexes](https://surrealdb.com/docs/surrealql/statements/define/indexes)

### Endpoint API
- Backend Swagger: http://192.168.1.200:5055/docs
- Frontend: http://192.168.1.200:8502

### Database
- SurrealDB: http://localhost:8000
- Namespace: `open_notebook`
- Database: `open_notebook`

---

## 🔄 Future Enhancements

### 1. Notebook Permissions
Aggiungere sistema di permessi per limitare accesso ai notebook:
```sql
-- User can access notebooks
CREATE TABLE user_notebook_access TYPE RELATION 
  IN user OUT notebook 
  SCHEMAFULL;
```

### 2. Notebook Sharing
Permettere condivisione source tra notebook con controllo granulare:
```sql
-- Source visibility settings
ALTER TABLE reference ADD shared BOOL DEFAULT true;
```

### 3. Notebook Analytics
Tracciare statistiche per notebook:
```sql
CREATE TABLE notebook_stats {
  notebook: record<notebook>,
  total_sources: int,
  total_searches: int,
  last_updated: datetime
};
```

### 4. Hierarchical Notebooks
Supportare notebook nested (progetti → sotto-progetti):
```sql
CREATE TABLE notebook_hierarchy TYPE RELATION
  IN notebook OUT notebook;
```

---

## ✅ Conclusioni

L'integrazione del filtro per notebook è:
- **Fattibile** - struttura database già pronta
- **Non invasiva** - backward compatible
- **Scalabile** - supporta multi-tenancy
- **Manutenibile** - modifiche concentrate in pochi punti

La roadmap è chiara e i rischi sono minimi. Si consiglia di procedere con implementazione graduale:
1. Database (foundation)
2. Backend (API layer)
3. Frontend (UX)
4. Testing & deployment

---

**Next Steps:**
1. Revisione tecnica con team
2. Prioritizzazione nel backlog
3. Assegnazione task
4. Implementazione fase 1 (database)
