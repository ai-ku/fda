#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <glib.h>
#include "procinfo.h"
#include "foreach.h"
#include "minialloc.h"
#include "token.h"
#include "sentence.h"
#include "heap.h"
#include "ngram.h"

/* Types */
typedef struct feat_s {
  guint train_cnt;
  guint output_cnt;
  double fscore0;
  double fscore1;
} *feat_t;

/* Functions */

static GHashTable *init_features(GPtrArray *sent, guint *bgcnt_ptr);
static guint init_train_count(GHashTable *feats, GPtrArray *sent);
static void init_feature_scores(gpointer key, gpointer val, gpointer dat);
static Heap init_sentence_heap(GHashTable *feat, GPtrArray *sent);
static gfloat sentence_score(Sentence s, GHashTable *feat);
static guint next_best_training_instance(Heap h, GPtrArray *sent, GHashTable *feat, gfloat *score_ptr, guint *niter_ptr);
static guint update_counts(GHashTable *feat, Sentence s);

/* Macros */

#define msg1 if(verbosity_level>=1)g_message
#define msg2 if(verbosity_level>=2)g_message

/* We often need to iterate through all ngrams (up to ngram_order) of a sentence. */
/* The macro below makes the rest of the code more readable. */
/* Note that the zeroth element of ngram/sentence gives the length. */
#define foreach_ngram(ngram, sentence)\
  for (guint32 *ngram = (sentence), *_last = ngram+((sentence)[0]), _save = ngram[0];\
       ngram <= _last; ngram[0] = _save, ngram++, _save = ngram[0])\
    for (ngram[0] = 1; (ngram[0] <= ngram_order) && (ngram+(ngram[0]) <= _last); ngram[0]++)
