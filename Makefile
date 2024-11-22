EXTENSION = llamasql
EXTVERSION = 0.1.0

PG_CONFIG = pg_config
PKG_CONFIG = pkg-config

MODULE_big = $(EXTENSION)

SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:.c=.o)

HEADERS = src/llamasql.h

DATA = $(wildcard sql/*--*.sql)

ifeq (no,$(shell $(PKG_CONFIG) llama || echo no))
$(warning libllama not registed with pkg-config, build might fail)
endif

LLAMA_CFLAGS = $(shell $(PKG_CONFIG) --cflags llama)
LLAMA_LDFLAGS = $(shell $(PKG_CONFIG) --libs llama)

PG_CFLAGS = -Wno-error=format-security -std=gnu99
PG_CPPFLAGS = $(LLAMA_CFLAGS)
PG_LDFLAGS = $(LLAMA_LDFLAGS)
SHLIB_LINK = -lllama

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# ---------- Test Models ----------
TEST_MODELS := qwen2 #qwen2-0_5b-instruct-q8_0 tinyllama-1.1b-1t-openorca.Q2_K phi-2.Q2_K
MODELS_PATHS := $(patsubst %,models/%.gguf,$(TEST_MODELS))
MODELS_TMP_PATHS := $(patsubst %,/tmp/%.gguf,$(TEST_MODELS))

models/phi-%.gguf:
	wget "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-$*.gguf?download=true" -O $@

models/tinyllama-%.gguf:
	wget "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-$*.gguf?download=true" -O $@

models/qwen2-%.gguf:
	wget "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-$*.gguf?download=true" -O $@

models/qwen2.gguf:
	wget "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q8_0.gguf?download=true" -O $@

/tmp/%.gguf: models/%.gguf
	cp $^ $@

installcheck: $(MODELS_TMP_PATHS)
	@echo "All models have been copied to /tmp"

# ---------- Examples ----------
EXAMPLES_SRCS = $(wildcard examples/*.c)
EXAMPLES_EXECS = $(patsubst examples/%.c, llama-%, $(EXAMPLES_SRCS))
llama-%: examples/%.c
	$(CC) $(LLAMA_CFLAGS) $< -o $@ $(LLAMA_LDFLAGS)

llama-examples: $(EXAMPLES_EXECS)
EXTRA_CLEAN = $(EXAMPLES_EXECS) dist

all: llama-examples

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

dev: clean all install installcheck

.DEFAULT: dev

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip main

# ---------- Docker ----------

PG_MAJOR ?= 17
LLAMA_VERSION ?= b4079

.PHONY: docker-build

docker-build:
	docker build --pull --no-cache --build-arg PG_MAJOR=$(PG_MAJOR) -t florents/llamasql:pg$(PG_MAJOR) -t florents/llamasql:$(EXTVERSION)-pg$(PG_MAJOR) .

.PHONY: docker-release

docker-release:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) -t florents/llamasql:pg$(PG_MAJOR) -t florents/llamasql:$(EXTVERSION)-pg$(PG_MAJOR) .