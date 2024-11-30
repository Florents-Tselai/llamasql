#ifndef PGLLAMASQL_H
#define PGLLAMASQL_H












/* cstring -> llama_model *
 * This should handle cache lookup as well
 */
// llama_model* load_llama_model_from_cstring(const char* path);
//
// #define PG_GETARG_LLAMA_MODEL(n) \
// ({ \
// llama_model_type* model_data = (llama_model_type*)PG_GETARG_POINTER(n); \
// const char* model_path = VARDATA(model_data); \
// load_llama_model_from_cstring(model_path); \
// })

/* ---------- llama_model metadata ---------- */



/* ---------- vocab / tokens---------- */





/* ---------- special tokens ---------- */



/*  ---------- prompt text* <--> char* ---------- */

/*
 * Usage:
 * int prompt_len;
 * char *prompt = PG_GETARG_LLAMA_TOKENS(n, n_input_tokens);
 */



/*  ---------- tokenization ---------- */





/* ---------- Variou Defaults ---------- */

#define BFRSZ 250

#endif