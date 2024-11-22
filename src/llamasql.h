#ifndef PGLLAMASQL_H
#define PGLLAMASQL_H
#include "postgres.h"
#include <utils/palloc.h>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/datetime.h"
#include "utils/date.h"
#include "utils/array.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <utils/guc.h>


#if PG_VERSION_NUM >= 160000
#include "varatt.h"

#endif

#if PG_VERSION_NUM < 130000
#define TYPALIGN_DOUBLE 'd'
#define TYPALIGN_INT 'i'
#endif

#include "llama.h"


/* ---------- missing typedefs from libllama ---------- */

typedef struct llama_context_params llama_context_params;
typedef struct llama_model_params llama_model_params;
typedef struct llama_model llama_model;

/* ---------- llama params <--> jsonb  ---------- */

llama_context_params    parse_context_params_from_jsonb(Jsonb *in_context_params);
llama_model_params      parse_model_params_from_jsonb(Jsonb *in_model_params);

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* ---------- llama_model_type ---------- */

/* Currently a llama_model_type is basically a text type that points to the .gguf model file */
typedef struct {
    int32   vl_len_;		            /* varlena header (do not touch directly!) */
    char    data[FLEXIBLE_ARRAY_MEMBER];
} llama_model_type;

PG_FUNCTION_INFO_V1(llama_model_in);
PG_FUNCTION_INFO_V1(llama_model_out);

/* cstring -> llama_model *
 * This should handle cache lookup as well
 */
llama_model* load_llama_model_from_cstring(const char* path);

#define PG_GETARG_LLAMA_MODEL(n) \
({ \
llama_model_type* model_data = (llama_model_type*)PG_GETARG_POINTER(n); \
const char* model_path = VARDATA(model_data); \
load_llama_model_from_cstring(model_path); \
})

/* ---------- llama_model metadata ---------- */

PG_FUNCTION_INFO_V1(pg_llama_model_desc);
PG_FUNCTION_INFO_V1(pg_llama_model_size);
PG_FUNCTION_INFO_V1(pg_llama_model_n_params);
PG_FUNCTION_INFO_V1(pg_llama_model_has_encoder);
PG_FUNCTION_INFO_V1(pg_llama_model_has_decoder);
PG_FUNCTION_INFO_V1(pg_llama_model_is_recurrent);

/* ---------- vocab / tokens---------- */

typedef int32 llama_token_type;

#define LLAMATOKENOID INT4OID

#define PG_GETARG_LLAMA_TOKEN(n)    PG_GETARG_INT32(n)
#define PG_RETURN_LLAMA_TOKEN(x)    PG_RETURN_INT32(x)

/*
 * Usage:
 * int n_input_tokens;
 * llama_token *input_tokens = PG_GETARG_LLAMA_TOKENS(n, n_input_tokens);
 */
#define PG_GETARG_LLAMA_TOKENS(n, n_input_tokens) ({ \
ArrayType *_input_array = PG_GETARG_ARRAYTYPE_P(n); \
if (ARR_NDIM(_input_array) != 1) { \
ereport(ERROR, \
(errcode(ERRCODE_INVALID_PARAMETER_VALUE), \
errmsg("Only one-dimensional arrays are supported"))); \
} \
(n_input_tokens) = ArrayGetNItems(ARR_NDIM(_input_array), ARR_DIMS(_input_array)); \
int32 *_array_data = (int32 *) ARR_DATA_PTR(_input_array); \
llama_token *_tokens = palloc((n_input_tokens) * sizeof(llama_token)); \
for (int _i = 0; _i < (n_input_tokens); _i++) { \
_tokens[_i] = (llama_token)_array_data[_i]; \
} \
_tokens; \
})

PG_FUNCTION_INFO_V1(pg_llama_token_text);
PG_FUNCTION_INFO_V1(pg_llama_token_score);

PG_FUNCTION_INFO_V1(pg_llama_token_is_eog);
PG_FUNCTION_INFO_V1(pg_llama_token_is_control);

/* ---------- special tokens ---------- */

PG_FUNCTION_INFO_V1(pg_llama_token_bos);
PG_FUNCTION_INFO_V1(pg_llama_token_eos);
PG_FUNCTION_INFO_V1(pg_llama_token_eot);
PG_FUNCTION_INFO_V1(pg_llama_token_cls);
PG_FUNCTION_INFO_V1(pg_llama_token_sep);
PG_FUNCTION_INFO_V1(pg_llama_token_nl);
PG_FUNCTION_INFO_V1(pg_llama_token_pad);

/*  ---------- prompt text* <--> char* ---------- */

/*
 * Usage:
 * int prompt_len;
 * char *prompt = PG_GETARG_LLAMA_TOKENS(n, n_input_tokens);
 */

#define PG_GETARG_PROMPT(n, prompt_len) ({ \
text *_text = PG_GETARG_TEXT_PP(n); \
char *_prompt = text_to_cstring(_text); \
(prompt_len) = VARSIZE_ANY_EXHDR(_text); \
_prompt; \
})

/*  ---------- tokenization ---------- */

PG_FUNCTION_INFO_V1(pg_llama_tokenize);
PG_FUNCTION_INFO_V1(pg_llama_detokenize);

/* ---------- Generate API ---------- */

PG_FUNCTION_INFO_V1(pg_llama_generate_from_text);
PG_FUNCTION_INFO_V1(pg_llama_generate_from_tokens);

/* ---------- Defaults ---------- */

#define BFRSZ 250

#endif