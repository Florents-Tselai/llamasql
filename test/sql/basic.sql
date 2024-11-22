/* ---------- llama_model_type ---------- */

SELECT '/tmp/qwen2.gguf'::llama_model;

/* ---------- llama_model metadata ---------- */
SELECT llama_model_desc('/tmp/qwen2.gguf');
SELECT pg_size_pretty(llama_model_size('/tmp/qwen2.gguf'::llama_model));
SELECT llama_model_n_params('/tmp/qwen2.gguf'::llama_model);
SELECT llama_model_has_encoder('/tmp/qwen2.gguf'::llama_model);
SELECT llama_model_has_decoder('/tmp/qwen2.gguf'::llama_model);
SELECT llama_model_is_recurrent('/tmp/qwen2.gguf'::llama_model);

/* ---------- llama_token_type ---------- */

SELECT 9707::llama_token; -- a llama_token is just an integer
SELECT '{785,3974,13876,38835,34208}'::llama_token[];

/* ---------- vocab / tokens---------- */

SELECT llama_token_text('/tmp/qwen2.gguf', 9707);

/* ---------- special tokens ---------- */
SELECT llama_token_bos('/tmp/qwen2.gguf');
SELECT llama_token_eos('/tmp/qwen2.gguf');
SELECT llama_token_eot('/tmp/qwen2.gguf');
SELECT llama_token_cls('/tmp/qwen2.gguf');
SELECT llama_token_sep('/tmp/qwen2.gguf');
SELECT llama_token_nl('/tmp/qwen2.gguf');
SELECT llama_token_pad('/tmp/qwen2.gguf');


/*  ---------- tokenization ---------- */

-- CLI: llama-tokenize -m /tmp/qwen2.gguf -p "Hello World" --ids
-- Also reproduce llama-cppy-python/tests/test_llama.py: test_real_model

SELECT llama_tokenize('/tmp/qwen2.gguf', 'Hello, world!', add_special => true, parse_special => true);
SELECT llama_tokenize('/tmp/qwen2.gguf', 'The quick brown fox jumps', add_special => true, parse_special => true);

SELECT llama_detokenize('/tmp/qwen2.gguf'::llama_model,
                        '{9707,11,1879,0}'::llama_token[],
                        remove_special => true, unparse_special => true);
SELECT llama_detokenize('/tmp/qwen2.gguf'::llama_model,
                        '{785,3974,13876,38835,34208}'::llama_token[],
                        remove_special => true, unparse_special => true);

/* ---------- Generate API ---------- */

SELECT llama_generate('/tmp/qwen2.gguf', 'The quick brown fox jumps');

SELECT llama_detokenize('/tmp/qwen2.gguf'::llama_model,
                        '{916,279,15678,5562}'::llama_token[],
                        remove_special => true, unparse_special => true);