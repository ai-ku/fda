#include "fda.h"

/* Default options */
guint verbosity_level = 1;
guint ngram_order = 3;
guint output_words = 100000;
double idf_exponent = 1.0;
double ngram_length_exponent = 1.0;
double decay_factor = 0.5;
double sentence_length_exponent = 1.0;
char *test_file1 = NULL;
char *test_file2 = NULL;
char *train_file1 = NULL;
char *train_file2 = NULL;

int main(int argc, char **argv) {
  g_message_init();
  int opt;
  while ((opt = getopt(argc, argv, "n:t:i:l:f:s:v:1:2:")) != -1) {
    switch (opt) {
    case 'n': ngram_order = atoi(optarg); break;
    case 't': output_words = atoi(optarg); break;
    case 'i': idf_exponent = atof(optarg); break;
    case 'l': ngram_length_exponent = atof(optarg); break;
    case 'f': decay_factor = atof(optarg); break;
    case 's': sentence_length_exponent = atof(optarg); break;
    case 'v': verbosity_level = atoi(optarg); break;
    case '1': test_file1 = optarg; break;
    case '2': test_file2 = optarg; break;
    default: g_error("Bad option -%c", opt); break;
    }
  }
  // optind is the first nonoption arg
  if (optind != argc - 2) g_error("Usage: fda [opts] file1 file2");
  train_file1 = argv[optind++];
  train_file2 = argv[optind++];
  
  msg2("Reading %s...", train_file1);
  GPtrArray *train1 = read_sentences(train_file1);
  msg2("Reading %s...", train_file2);
  GPtrArray *train2 = read_sentences(train_file2);
  g_assert(train1->len == train2->len);

  msg2("Reading %s...", test_file1);
  GPtrArray *test1 = read_sentences(test_file1);
  msg2("Reading %s...", test_file2);
  GPtrArray *test2 = read_sentences(test_file2);
  g_assert(test1->len == test2->len);

  guint bigram_cnt1, bigram_cnt2;
  msg2("init_features1");
  GHashTable *features1 = init_features(test1, &bigram_cnt1);
  msg2("init_features2");
  GHashTable *features2 = init_features(test2, &bigram_cnt2);
  msg2("init_train_count");
  guint train_size1 = (idf_exponent == 0) ? 0 : init_train_count(features1, train1);
  msg2("init_feature_scores");
  g_hash_table_foreach(features1, init_feature_scores, &train_size1);
  msg2("init_sentence_heap");
  Heap heap = init_sentence_heap(features1, train1);
  guint nword1 = 0;
  guint nword2 = 0;
  guint bigram_match1 = 0;
  guint bigram_match2 = 0;
  
  msg2("Writing...");
  while (1) {
    if (heap_size(heap) == 0) break;
    gfloat best_score = 0;
    guint nscore = 0;
    guint best_sentence = next_best_training_instance(heap, train1, features1, &best_score, &nscore);
    Sentence s1 = g_ptr_array_index(train1, best_sentence);
    Sentence s2 = g_ptr_array_index(train2, best_sentence);
    nword1 += sentence_size(s1);
    nword2 += sentence_size(s2);
    bigram_match1 += update_counts(features1, s1);
    bigram_match2 += update_counts(features2, s2);
    print_sentence(s1); putchar('\t'); print_sentence(s2);
    printf("\t%g\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", best_score, nscore, nword1, 
	   nword2, bigram_cnt1, bigram_cnt2, bigram_match1, bigram_match2);
    if (nword1 >= output_words) break;
  }
  minialloc_free_all();
  msg1("-1%s -2%s -f%g -i%g -l%g -n%d -s%g -t%d -v%d %s %s\t%d\t%d\t%d\t%d\t%d\t%d",
       test_file1, test_file2, decay_factor, idf_exponent, ngram_length_exponent,
       ngram_order, sentence_length_exponent, output_words, verbosity_level,
       train_file1, train_file2, nword1, nword2, bigram_cnt1, bigram_cnt2, 
       bigram_match1, bigram_match2);
}

static GHashTable *init_features(GPtrArray *sent, guint *bigram_cnt) {
  GHashTable *feats = g_hash_table_new(ngram_hash, ngram_equal);
  *bigram_cnt = 0;
  for (int si = 0; si < sent->len; si++) {
    Sentence s = g_ptr_array_index(sent, si);
    foreach_ngram(ng, s) {
      if (g_hash_table_lookup(feats, ng) == NULL) {
	feat_t f = minialloc(sizeof(struct feat_s));
	f->output_cnt = 0; f->train_cnt = 0; f->fscore0 = 0; f->fscore1 = 0;
	g_hash_table_insert(feats, ngram_dup(ng), f);
	if (ngram_size(ng) == 2) (*bigram_cnt)++;
      }
    }
  }
  return feats;
}

static guint init_train_count(GHashTable *feats, GPtrArray *sent) {
  guint nwords = 0;
  for (int si = 0; si < sent->len; si++) {
    Sentence s = g_ptr_array_index(sent, si);
    nwords += sentence_size(s);
    foreach_ngram(ng, s) {
      feat_t f = g_hash_table_lookup(feats, ng);
      if (f != NULL) f->train_cnt++;
    }
  }
  return nwords;
}

static void init_feature_scores(gpointer key, gpointer val, gpointer dat) {
  Ngram ng = key;
  feat_t f = val;
  f->fscore0 = 1;
  if (ngram_length_exponent != 0) {
    double x = (double) ngram_size(ng);
    if (ngram_length_exponent != 1) 
      x = pow(x, ngram_length_exponent);
    f->fscore0 *= x;
  }
  if (idf_exponent != 0) {
    guint *train_size1 = dat;
    guint fcnt = f->train_cnt;
    if (fcnt == 0) fcnt = 1;
    double idf = -log((double) fcnt / (double) (*train_size1));
    if (idf_exponent != 1) idf = pow(idf, idf_exponent);
    f->fscore0 *= idf;
  }
  f->fscore1 = f->fscore0;
}

static Heap init_sentence_heap(GHashTable *feat, GPtrArray *sent) {
  Heap heap = minialloc(sizeof(Hpair) * (1 + sent->len));
  heap_size(heap) = 0;
  for (guint si = 0; si < sent->len; si++) {
    Sentence s = g_ptr_array_index(sent, si);
    gfloat sscore = (gfloat) sentence_score(s, feat);
    heap_insert_max(heap, si, sscore);
  }
  return heap;
}

static gfloat sentence_score(Sentence s, GHashTable *feat) {
  gfloat score = 0;
  foreach_ngram(ng, s) {
    feat_t f = g_hash_table_lookup(feat, ng);
    if (f != NULL) score += f->fscore1;
  }
  if ((sentence_length_exponent != 0) && (sentence_size(s) > 0)) {
    double x = (double) sentence_size(s);
    if (sentence_length_exponent != 1)
      x = pow(x, sentence_length_exponent);
    score /= x;
  }
  return score;
}

static guint next_best_training_instance(Heap h, GPtrArray *sent, GHashTable *feat, gfloat *score_ptr, guint *niter_ptr) {
  g_assert(heap_size(h) > 0);
  guint best_sentence = 0;
  gfloat best_score = 0;
  guint niter = 0;
  while (1) {
    niter++;
    best_sentence = heap_top(h).key;
    best_score = sentence_score(g_ptr_array_index(sent, best_sentence), feat);
    heap_delete_max(h);
    if (heap_size(h) == 0) break;
    if (best_score / heap_top(h).val >= 1.0 - G_MINFLOAT) break;
    heap_insert_max(h, best_sentence, best_score);
  }
  *score_ptr = best_score;
  *niter_ptr = niter;
  return best_sentence;
}

static guint update_counts(GHashTable *feat, Sentence s) {
  guint bigram_match = 0;
  foreach_ngram(ng, s) {
    feat_t f = g_hash_table_lookup(feat, ng);
    if (f != NULL) {
      if ((ngram_size(ng) == 2) && (f->output_cnt == 0)) bigram_match++;
      f->output_cnt++;
      if (decay_factor >= 1) {
	f->fscore1 = f->fscore0 / (1.0 + f->output_cnt);
      } else if (decay_factor > 0) {
	f->fscore1 = f->fscore0 * pow(decay_factor, f->output_cnt);
      }
    }
  }
  return bigram_match;
}

