/* TYPE=DOMAIN: context_params_j */

CREATE DOMAIN llama_context_params_j AS jsonb
    CHECK (
        -- Example: Ensure JSON contains specific keys with allowed values
        (jsonb_typeof(VALUE) = 'object') AND
        VALUE ? 'max_connections' AND (VALUE->>'max_connections')::int >= 1
    );


/* TYPE: llama_model */

-- SEE https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
CREATE TYPE llama_model;

CREATE FUNCTION llama_model_in(cstring) RETURNS llama_model IMMUTABLE
    STRICT
    LANGUAGE C
AS
'MODULE_PATHNAME';

CREATE FUNCTION llama_model_out(llama_model) RETURNS cstring IMMUTABLE
    STRICT
    LANGUAGE C
AS
'MODULE_PATHNAME';

CREATE TYPE llama_model (
    INTERNALLENGTH = -1,
    INPUT = llama_model_in,
    OUTPUT = llama_model_out,
    STORAGE = extended
);

CREATE CAST (llama_model AS text) WITH INOUT AS ASSIGNMENT;
CREATE CAST (text AS llama_model) WITH INOUT AS ASSIGNMENT;


CREATE DOMAIN llama_token AS INTEGER;
CREATE DOMAIN llama_pos AS INTEGER;
CREATE DOMAIN llama_seq_id AS INTEGER;

/* llama_model metadata */

CREATE FUNCTION llama_model_desc(llama_model) RETURNS text IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_desc';
CREATE FUNCTION llama_model_size(llama_model) RETURNS bigint IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_size';
CREATE FUNCTION llama_model_n_params(llama_model) RETURNS bigint IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_n_params';
CREATE FUNCTION llama_model_has_encoder(llama_model) RETURNS boolean IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_has_encoder';
CREATE FUNCTION llama_model_has_decoder(llama_model) RETURNS boolean IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_has_decoder';
CREATE FUNCTION llama_model_is_recurrent(llama_model) RETURNS boolean IMMUTABLE STRICT LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_model_is_recurrent';

/* vocab / tokens */
CREATE FUNCTION llama_token_text(llama_model, llama_token) RETURNS text IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_text';
CREATE FUNCTION llama_token_score(llama_model, llama_token) RETURNS float IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_score';

CREATE FUNCTION llama_token_is_eog(llama_model, llama_token) RETURNS bool IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_is_eog';
CREATE FUNCTION llama_token_is_control(llama_model, llama_token) RETURNS bool IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_is_control';

/* Special tokens */
CREATE FUNCTION llama_token_bos(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_bos';
CREATE FUNCTION llama_token_eos(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_eos';
CREATE FUNCTION llama_token_eot(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_eot';
CREATE FUNCTION llama_token_cls(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_cls';
CREATE FUNCTION llama_token_sep(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_sep';
CREATE FUNCTION llama_token_nl(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_nl';
CREATE FUNCTION llama_token_pad(llama_model) RETURNS llama_token IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_token_pad';


/* API: tokenization */
CREATE FUNCTION llama_tokenize(model llama_model,
                               prompt text,
                               add_special boolean default true,
                               parse_special boolean default false,
                               context_params jsonb default null,
                               model_params jsonb default null)
    returns llama_token[] IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_tokenize';

CREATE FUNCTION llama_detokenize(model llama_model,
                                tokens llama_token[],
                                remove_special boolean default true,
                                unparse_special boolean default false,
                                context_params jsonb default null,
                                model_params jsonb default null)
    returns text IMMUTABLE LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_detokenize';

/* TYPE: llama_chat_message */

CREATE TYPE llama_chat_message AS (
    role_ text,
    content text
);


/* ---------- Generate API ---------- */

CREATE FUNCTION llama_generate_from_text(model llama_model,
                               prompt text,
                               context_params jsonb default null,
                               model_params jsonb default null)
    returns llama_token[] LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_generate_from_text';

CREATE FUNCTION llama_generate_from_tokens(model llama_model,
                                           tokens llama_token[],
                                         context_params jsonb default null,
                                         model_params jsonb default null)
    returns llama_token[] LANGUAGE C AS 'MODULE_PATHNAME', 'pg_llama_generate_from_tokens';