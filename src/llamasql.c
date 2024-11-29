#include "llamasql.h"

llama_context_params parse_context_params_from_jsonb(Jsonb *in_context_params)
{
    JsonbIterator *it;
    JsonbValue v;
    int r;
    struct llama_context_params context_params = llama_context_default_params();
    /* Initialize iterator */
    it = JsonbIteratorInit(&in_context_params->root);

    /* Iterate over the JSONB key-value pairs */
    while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
    {
        if (r == WJB_KEY)
        {
            char *key = pnstrdup(v.val.string.val, v.val.string.len);

            r = JsonbIteratorNext(&it, &v, false);

            if (r == WJB_VALUE)
            {
                /* Integer params */
                if (strcmp(key, "n_ctx") == 0 && v.type == jbvNumeric)
                    context_params.n_ctx = DatumGetUInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "n_batch") == 0 && v.type == jbvNumeric)
                    context_params.n_batch = DatumGetUInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "n_ubatch") == 0 && v.type == jbvNumeric)
                    context_params.n_ubatch = DatumGetUInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "n_seq_max") == 0 && v.type == jbvNumeric)
                    context_params.n_seq_max = DatumGetUInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "n_threads") == 0 && v.type == jbvNumeric)
                    context_params.n_threads = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "n_threads_batch") == 0 && v.type == jbvNumeric)
                    context_params.n_threads_batch = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                /* Enum params */
                else if (strcmp(key, "rope_scaling_type") == 0 && v.type == jbvNumeric)
                    context_params.rope_scaling_type = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));
                else if (strcmp(key, "pooling_type") == 0 && v.type == jbvNumeric)
                    context_params.pooling_type = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));
                else if (strcmp(key, "attention_type") == 0 && v.type == jbvNumeric)
                    context_params.attention_type = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                /* float params */
                else if (strcmp(key, "rope_freq_base") == 0 && v.type == jbvNumeric)
                    context_params.rope_freq_base = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "rope_freq_scale") == 0 && v.type == jbvNumeric)
                    context_params.rope_freq_scale = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "yarn_ext_factor") == 0 && v.type == jbvNumeric)
                    context_params.yarn_ext_factor = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "yarn_attn_factor") == 0 && v.type == jbvNumeric)
                    context_params.yarn_attn_factor = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "yarn_beta_fast") == 0 && v.type == jbvNumeric)
                    context_params.yarn_beta_fast = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "yarn_beta_slow") == 0 && v.type == jbvNumeric)
                    context_params.yarn_beta_slow = DatumGetFloat4(DirectFunctionCall1(numeric_float4, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "yarn_orig_ctx") == 0 && v.type == jbvNumeric)
                    context_params.yarn_orig_ctx = DatumGetUInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                /* bool params */
                else if (strcmp(key, "logits_all") == 0 && v.type == jbvBool)
                    context_params.logits_all = v.val.boolean;

                else if (strcmp(key, "embeddings") == 0 && v.type == jbvBool)
                    context_params.embeddings = v.val.boolean;

                else if (strcmp(key, "offload_kqv") == 0 && v.type == jbvBool)
                    context_params.offload_kqv = v.val.boolean;

                else if (strcmp(key, "flash_attn") == 0 && v.type == jbvBool)
                    context_params.flash_attn = v.val.boolean;

                else if (strcmp(key, "no_perf") == 0 && v.type == jbvBool)
                    context_params.no_perf = v.val.boolean;
                else
                    elog(ERROR, "Unsupported llama_context_params option %s", key);
            }
            pfree(key);
        }
    }
    return context_params;
}
llama_model_params parse_model_params_from_jsonb(Jsonb *in_model_params)
{
    JsonbIterator *it;
    JsonbValue v;
    int r;
    struct llama_model_params model_params = llama_model_default_params();

    /* Initialize iterator */
    it = JsonbIteratorInit(&in_model_params->root);

    /* Iterate over the JSONB key-value pairs */
    while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
    {
        if (r == WJB_KEY)
        {
            char *key = pnstrdup(v.val.string.val, v.val.string.len);

            r = JsonbIteratorNext(&it, &v, false);

            if (r == WJB_VALUE)
            {
                if (strcmp(key, "n_gpu_layers") == 0 && v.type == jbvNumeric)
                    model_params.n_gpu_layers = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "split_mode") == 0 && v.type == jbvNumeric)
                    model_params.split_mode = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "main_gpu") == 0 && v.type == jbvNumeric)
                    model_params.main_gpu = DatumGetInt32(DirectFunctionCall1(numeric_int8, NumericGetDatum(v.val.numeric)));

                else if (strcmp(key, "tensor_split") == 0)
                    elog(ERROR, "Handling of %s is not implemented yet", key);

                else if (strcmp(key, "rpc_servers") == 0)
                    elog(ERROR, "Handling of %s is not implemented yet", key);

                /* bool params */
                else if (strcmp(key, "vocab_only") == 0 && v.type == jbvBool)
                    model_params.vocab_only = v.val.boolean;

                else if (strcmp(key, "use_mmap") == 0 && v.type == jbvBool)
                    model_params.use_mmap = v.val.boolean;

                else if (strcmp(key, "use_mlock") == 0 && v.type == jbvBool)
                    model_params.use_mlock = v.val.boolean;

                else if (strcmp(key, "check_tensors") == 0 && v.type == jbvBool)
                    model_params.check_tensors = v.val.boolean;

                else
                    elog(ERROR, "Unsupported llama_context_params option %s", key);
            }
            pfree(key);
        }
    }
    return model_params;
}


#include "fmgr.h"
#include "funcapi.h"
#include "utils/jsonb.h"
#include "storage/lwlock.h"
#include "storage/dsm_registry.h"
#include "storage/shmem.h"


#define LLAMASQL_MAX_MODEL_FILENAME NAMEDATALEN
#define LLAMASQL_MAX_MODEL_FILE_SIZE (1024 * 1024 * 1024) /* 1GB */
#define LLAMASQL_SHMEM_NAME "llamasql"

/*
 * FTTB, for simplicity we only cache *one* model.
 * In the future we may want to have a few slots available.
 *
 */
typedef struct LLamaSQLSharedState
{
    LWLock          lock;

    char    path[LLAMASQL_MAX_MODEL_FILENAME];              /* filepath to be used as a key*/
    int32	vl_len_;                        /* number of bytes in the file */
    char    data[FLEXIBLE_ARRAY_MEMBER];    /* actual file bytes */

} LLamaSQLSharedState;

/* Pointers to shared-memory */
static LLamaSQLSharedState *g_state = NULL;

static void
llsql_init_state(void *ptr)
{
    LLamaSQLSharedState *state = (LLamaSQLSharedState *) ptr;
    LWLockInitialize(&state->lock, LWLockNewTrancheId());

}

static bool
llamasql_init_shmem(void)
{
    bool found;
    g_state = GetNamedDSMSegment(LLAMASQL_SHMEM_NAME,
                                   offsetof(LLamaSQLSharedState, data) + LLAMASQL_MAX_MODEL_FILE_SIZE,
                                   llsql_init_state,
                                   &found);
    LWLockRegisterTranche(g_state->lock.tranche, LLAMASQL_SHMEM_NAME);

    return found;
}

void
_PG_init(void)
{
    llama_backend_init();
    llamasql_init_shmem();
}

llama_model* load_llama_model_from_cstring(const char* path)
{
    llama_model *result = NULL;
    llama_model_params model_params = llama_model_default_params();

    char *model_swap_name = "modelswap.gguf";

    LWLockAcquire(&g_state->lock, LW_EXCLUSIVE);
    bool model_is_cached = strcmp(g_state->path, path) == 0;
    LWLockRelease(&g_state->lock);

    if (!model_is_cached)
    {
        Datum model_file = DirectFunctionCall1(pg_read_binary_file_all, PointerGetDatum(cstring_to_text(path)));
        LWLockAcquire(&g_state->lock, LW_EXCLUSIVE);

        memset(g_state->path, 0, LLAMASQL_MAX_MODEL_FILENAME);
        strncpy(g_state->path, path, LLAMASQL_MAX_MODEL_FILENAME-1);
        g_state->path[LLAMASQL_MAX_MODEL_FILENAME-1] = '\0';
        g_state->vl_len_ = VARSIZE_ANY_EXHDR(model_file);
        memcpy(g_state->data, VARDATA_ANY(model_file), g_state->vl_len_);


        printf("shared cache: path=%s\tvl_len=%d\n", g_state->path, g_state->vl_len_);

        FILE *fd = AllocateFile(model_swap_name, PG_BINARY_W);
        if (!fd) elog(ERROR, "Failed to open %s for writing\n", model_swap_name);

        /* Write the binary data */
        if (fwrite(g_state->data, g_state->vl_len_, 1, fd) != 1)
        {
            FreeFile(fd);
            unlink(model_swap_name);
            elog(ERROR, "Failed to write binary data to %s \n", model_swap_name);

        }

        LWLockRelease(&g_state->lock);

        /* Flush the file to ensure all data is written */
        if (fflush(fd) != 0)
        {
            FreeFile(fd);
            unlink(model_swap_name);
            elog(ERROR, "Failed to flush %s", model_swap_name);
        }

        struct stat st;
        if (fstat(fileno(fd), &st) == 0)
        {
            printf("File size of %s: %ld bytes", "modelcopy.gguf", st.st_size);
        }
        else
        {
            printf("Failed to get file size for %s", "modelcopy.gguf");
        }

        result = llama_load_model_from_file(model_swap_name, model_params);

        FreeFile(fd);
        unlink(model_swap_name);

        if (result == NULL) elog(ERROR, "Failed to load Llama model from path: %s", path);
    }
    else /* model is cached */
    {
        printf("model is cached\n");
        LWLockAcquire(&g_state->lock, LW_EXCLUSIVE);
        printf("shared cache: path=%s\tvl_len=%d\n", g_state->path, g_state->vl_len_);
        model_swap_name = "/tmp/cacheswap.gguf";

        FILE *fd = AllocateFile(model_swap_name, PG_BINARY_W);
        if (!fd) elog(ERROR, "Failed to open %s for writing\n", model_swap_name);

        /* Write the binary data */
        if (fwrite(g_state->data, g_state->vl_len_, 1, fd) != 1)
        {
            FreeFile(fd);
            unlink(model_swap_name);
            elog(ERROR, "Failed to write binary data to %s \n", model_swap_name);

        }

        /* Flush the file to ensure all data is written */
        if (fflush(fd) != 0)
        {
            FreeFile(fd);
            unlink(model_swap_name);
            elog(ERROR, "Failed to flush %s", model_swap_name);
        }

        struct stat st;
        if (fstat(fileno(fd), &st) == 0)
        {
            printf("File size of %s: %ld bytes", "modelcopy.gguf", st.st_size);
        }
        else
        {
            printf("Failed to get file size for %s", "modelcopy.gguf");
        }

        result = llama_load_model_from_file(model_swap_name, model_params);

        LWLockRelease(&g_state->lock);

        FreeFile(fd);
        unlink(model_swap_name);

    }

    return result;
}

/* ---------- llama_model_type ---------- */

Datum
llama_model_in(PG_FUNCTION_ARGS)
{
    const char* model_path = PG_GETARG_CSTRING(0);
    llama_model_type* result;
    size_t path_length = strlen(model_path) + 1;

    result = (llama_model_type*)palloc(VARHDRSZ + path_length);
    SET_VARSIZE(result, VARHDRSZ + path_length);
    memcpy(VARDATA(result), model_path, path_length);

    PG_RETURN_POINTER(result);
}

Datum
llama_model_out(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    char desc_buf[BFRSZ];
    char* result;

    if (llama_model_desc(model, desc_buf, sizeof(desc_buf)) < 0)
        elog(ERROR, "Failed to retrieve model description");

    result = pstrdup(desc_buf);

    llama_free_model(model);

    PG_RETURN_CSTRING(result);
}


PG_FUNCTION_INFO_V1(llama_model_cache_add);
Datum
llama_model_cache_add(PG_FUNCTION_ARGS)
{
    Datum arg0 = PG_GETARG_DATUM(0);
    char *path = VARDATA_ANY(arg0);
    int32 path_len = VARSIZE_ANY_EXHDR(arg0);
    Datum file_bytes;
    int32 file_bytes_len;
    bytea *result;

    /* Initialize shared memory if not already done */
    //

    /* Check if path fits within the allocated space */
    if (path_len >= LLAMASQL_MAX_MODEL_FILENAME) {
        ereport(ERROR,
                (errmsg("Path length exceeds allowed limit of %d", LLAMASQL_MAX_MODEL_FILENAME)));
    }

    file_bytes = DirectFunctionCall1(pg_read_binary_file_all, arg0);
    file_bytes_len = VARSIZE_ANY_EXHDR(file_bytes);

    /* Copy path into shared state */
    memcpy(g_state->path, path, path_len);
    g_state->path[path_len] = '\0';
    g_state->vl_len_ = path_len;
    memcpy(g_state->data, VARDATA_ANY(file_bytes), file_bytes_len);

    result = palloc(VARHDRSZ + file_bytes_len);
    SET_VARSIZE(result, VARHDRSZ + file_bytes_len);
    memcpy(VARDATA(result), VARDATA_ANY(file_bytes), file_bytes_len);

    /* Return a dummy value (e.g., true) */
    PG_RETURN_BYTEA_P(result);
}

/* ---------- llama_model metadata ---------- */

Datum
pg_llama_model_desc(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    char desc_buf[BFRSZ];
    char* result;

    if (llama_model_desc(model, desc_buf, sizeof(desc_buf)) < 0)
        elog(ERROR, "Failed to retrieve model description");

    result = pstrdup(desc_buf);
    llama_free_model(model);
    PG_RETURN_TEXT_P(cstring_to_text(result));
}

Datum
pg_llama_model_size(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    uint64_t result = llama_model_size(model);

    llama_free_model(model);
    PG_RETURN_INT64(result);
}

Datum
pg_llama_model_n_params(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    uint64_t result = llama_model_n_params(model);

    llama_free_model(model);

    PG_RETURN_INT64(result);
}

Datum
pg_llama_model_has_encoder(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    bool result = llama_model_has_encoder(model);

    llama_free_model(model);

    PG_RETURN_BOOL(result);
}

Datum
pg_llama_model_has_decoder(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    bool result = llama_model_has_decoder(model);

    llama_free_model(model);

    PG_RETURN_INT64(result);
}

Datum
pg_llama_model_is_recurrent(PG_FUNCTION_ARGS)
{
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);

    bool result = llama_model_is_recurrent(model);

    llama_free_model(model);

    PG_RETURN_INT64(result);
}

/* ---------- vocab / tokens---------- */

Datum
pg_llama_token_text(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);
    llama_token t       = PG_GETARG_LLAMA_TOKEN(1);

    const char* result = llama_token_get_text(model, t);

    llama_free_model(model);
    PG_RETURN_TEXT_P(cstring_to_text(result));
}

Datum
pg_llama_token_score(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);
    llama_token t       = PG_GETARG_LLAMA_TOKEN(1);

    float result = llama_token_get_score(model, t);

    llama_free_model(model);
    PG_RETURN_FLOAT8(result);
}

Datum
pg_llama_token_is_eog(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);
    llama_token t       = PG_GETARG_LLAMA_TOKEN(1);

    bool result = llama_token_is_eog(model, t);

    llama_free_model(model);
    PG_RETURN_BOOL(result);
}

Datum
pg_llama_token_is_control(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);
    llama_token t       = PG_GETARG_LLAMA_TOKEN(1);

    bool result = llama_token_is_control(model, t);

    llama_free_model(model);
    PG_RETURN_BOOL(result);
}

/* ---------- special tokens ---------- */

Datum
pg_llama_token_bos(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_bos(model);

    llama_free_model(model);
    PG_RETURN_BOOL(result);
}

Datum
pg_llama_token_eos(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_eos(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}

Datum
pg_llama_token_eot(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_eot(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}

Datum
pg_llama_token_cls(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_cls(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}


Datum
pg_llama_token_sep(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_sep(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}

Datum
pg_llama_token_nl(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_nl(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}


Datum
pg_llama_token_pad(PG_FUNCTION_ARGS)
{
    llama_model *model  = PG_GETARG_LLAMA_MODEL(0);

    llama_token result = llama_token_pad(model);

    llama_free_model(model);
    PG_RETURN_LLAMA_TOKEN(result);
}

/*  ---------- tokenization ---------- */

Datum
pg_llama_tokenize(PG_FUNCTION_ARGS)
{
    llama_model *model                          = PG_GETARG_LLAMA_MODEL(0);
    int prompt_len;
    char *prompt = PG_GETARG_PROMPT(1, prompt_len);
    bool    add_special                         = PG_GETARG_BOOL(2);
    bool    parse_special                       = PG_GETARG_BOOL(3);
    struct llama_context_params context_params  = PG_GETARG_LLAMA_CONTEXT_PARAMS(4);
    struct llama_model_params model_params      = PG_GETARG_LLAMA_MODEL_PARAMS(5);


    int n_tokens = prompt_len + 2 * add_special;
    llama_token* tokens = palloc(n_tokens * sizeof(llama_token));

    n_tokens = llama_tokenize(model, prompt, prompt_len, tokens, n_tokens, add_special, parse_special);

    Datum* datums = palloc(n_tokens * sizeof(Datum));
    for (int i = 0; i < n_tokens; i++)
    {
        datums[i] = Int32GetDatum(tokens[i]);
    }

    ArrayType* array = construct_array(
        datums,
        n_tokens,
        INT4OID,
        sizeof(int32),
        true, // int32 is pass-by-value in PostgreSQL
        'i' // int32 is 4-byte aligned
    );

    pfree(tokens);
    pfree(datums);
    llama_free_model(model);

    PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_llama_detokenize(PG_FUNCTION_ARGS)
{
    llama_model *model                          = PG_GETARG_LLAMA_MODEL(0);
    int n_input_tokens;
    llama_token *input_tokens                   = PG_GETARG_LLAMA_TOKENS(1, n_input_tokens);
    bool remove_special                         = PG_GETARG_BOOL(2);
    bool unparse_special                        = PG_GETARG_BOOL(3);
    struct llama_context_params context_params  = PG_GETARG_LLAMA_CONTEXT_PARAMS(4);
    struct llama_model_params model_params      = PG_GETARG_LLAMA_MODEL_PARAMS(5);

    int32_t buffer_size = BFRSZ;
    char *text = palloc(buffer_size);
    int32_t n_chars;

    while (true) {
        n_chars = llama_detokenize(model, input_tokens, n_input_tokens, text, buffer_size, remove_special, unparse_special);
        if (n_chars < 0) {
            buffer_size = -n_chars;
            text = repalloc(text, buffer_size);
        } else {
            break;
        }
    }

    text[n_chars] = '\0';
    text = repalloc(text, n_chars + 1);

    pfree(input_tokens);
    llama_free_model(model);

    PG_RETURN_TEXT_P(cstring_to_text(text));
}

Datum
pg_llama_generate_from_text(PG_FUNCTION_ARGS)
{
    const int num_tokens_to_generate = 4;

    llama_model *model = PG_GETARG_LLAMA_MODEL(0);
    int prompt_len;
    char *prompt = PG_GETARG_PROMPT(1, prompt_len);

    struct llama_context_params context_params = PG_GETARG_LLAMA_CONTEXT_PARAMS(2);
    struct llama_model_params model_params = PG_GETARG_LLAMA_MODEL_PARAMS(3);

    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_params.check_tensors = false;

    context_params.n_ctx = 16;
    context_params.n_batch = 16;
    context_params.n_ubatch = 16;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.logits_all = false;
    context_params.flash_attn = true;

    struct llama_context *context = llama_new_context_with_model(model, context_params);

    const int n_prompt = -llama_tokenize(model, prompt, prompt_len, NULL, 0, true, true);
    llama_token *tokens = malloc(n_prompt * sizeof(llama_token));
    int n_tokens = llama_tokenize(model, prompt, prompt_len, tokens, n_prompt, true, true);

    if (n_tokens < 0) {
        free(tokens);
        ereport(ERROR, (errmsg("Failed to tokenize the prompt")));
    }

    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    struct llama_sampler *sampler = llama_sampler_chain_init(sampler_params);

    if (!sampler) {
        free(tokens);
        ereport(ERROR, (errmsg("Failed to initialize sampler chain")));
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1337));

    Datum *result_tokens = palloc(num_tokens_to_generate * sizeof(Datum));

    for (int i = 0; i < num_tokens_to_generate; i++) {
        struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);

        if (llama_decode(context, batch) < 0) {
            free(tokens);
            ereport(ERROR, (errmsg("Failed to encode the batch")));
        }

        llama_token token_id = llama_sampler_sample(sampler, context, -1);
        result_tokens[i] = Int32GetDatum((int32)token_id);

        tokens[0] = token_id;
        n_tokens = 1;
    }

    ArrayType *array = construct_array(
        result_tokens,
        num_tokens_to_generate,
        LLAMATOKENOID,
        sizeof(llama_token_type),
        true,
        'i'
    );

    free(tokens);
    pfree(result_tokens);
    llama_free_model(model);

    PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_llama_generate_from_tokens(PG_FUNCTION_ARGS)
{
    const int num_tokens_to_generate = 4;

    // Get the model and input tokens from PostgreSQL arguments
    llama_model *model = PG_GETARG_LLAMA_MODEL(0);
    int n_input_tokens;
    const llama_token *input_tokens = PG_GETARG_LLAMA_TOKENS(1, n_input_tokens);

    struct llama_context_params context_params = PG_GETARG_LLAMA_CONTEXT_PARAMS(2);
    struct llama_model_params model_params = PG_GETARG_LLAMA_MODEL_PARAMS(3);

    // Configure model and context parameters
    model_params.use_mmap = false;
    model_params.use_mlock = false;
    model_params.check_tensors = false;

    context_params.n_ctx = 16;
    context_params.n_batch = 16;
    context_params.n_ubatch = 16;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.logits_all = false;
    context_params.flash_attn = true;

    // Create a new context
    struct llama_context *context = llama_new_context_with_model(model, context_params);

    // Initialize the sampler
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    struct llama_sampler *sampler = llama_sampler_chain_init(sampler_params);

    if (!sampler) {
        ereport(ERROR, (errmsg("Failed to initialize sampler chain")));
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1337));

    Datum *result_tokens = palloc(num_tokens_to_generate * sizeof(Datum));

    // Copy input tokens into a dynamic buffer for modification
    llama_token *current_tokens = malloc((n_input_tokens + 1) * sizeof(llama_token));
    memcpy(current_tokens, input_tokens, n_input_tokens * sizeof(llama_token));
    int n_tokens = n_input_tokens;

    for (int i = 0; i < num_tokens_to_generate; i++) {
        // Create a batch with the current tokens
        struct llama_batch batch = llama_batch_get_one(current_tokens, n_tokens);

        // Decode the batch
        if (llama_decode(context, batch) < 0) {
            free(current_tokens);
            ereport(ERROR, (errmsg("Failed to encode the batch")));
        }

        // Sample the next token
        llama_token token_id = llama_sampler_sample(sampler, context, -1);
        result_tokens[i] = Int32GetDatum((int32)token_id);

        // Update the current tokens buffer with the newly generated token
        current_tokens[0] = token_id;
        n_tokens = 1;
    }

    // Construct the resulting array
    ArrayType *array = construct_array(
        result_tokens,
        num_tokens_to_generate,
        LLAMATOKENOID,
        sizeof(llama_token_type),
        true,
        'i'
    );

    // Free allocated memory
    free(current_tokens);
    pfree(result_tokens);
    llama_free_model(model);

    PG_RETURN_ARRAYTYPE_P(array);
}
