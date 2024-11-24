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


/* ---------- MODEL CACHE ----------*/

PGDLLEXPORT void _PG_init(void);

void
_PG_init(void)
{
    llama_backend_init();

}

llama_model* load_llama_model_from_cstring(const char* path)
{
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(path, model_params);
    if (model == NULL)
        elog(ERROR, "llama_load_model_from_file failed");
    return model;
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
