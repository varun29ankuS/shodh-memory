.PHONY: build run test check clean build-tui run-tui

# Detect macOS and Homebrew LLVM to provide a smoother build experience
# without requiring gnarly manual exports for LIBCLANG_PATH.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Find LLVM from Homebrew (checks version 21, then default llvm)
    LLVM_PREFIX := $(shell brew --prefix llvm@21 2>/dev/null || brew --prefix llvm 2>/dev/null || brew --prefix llvm@14 2>/dev/null || echo "")
    ifneq ($(LLVM_PREFIX),)
        export LIBCLANG_PATH := $(LLVM_PREFIX)/lib
        export DYLD_FALLBACK_LIBRARY_PATH := $(LLVM_PREFIX)/lib:$(DYLD_FALLBACK_LIBRARY_PATH)
    endif
endif

build:
	cargo build

run:
	cargo run

test:
	cargo test

check:
	cargo check

clean:
	cargo clean
	cargo clean --manifest-path tui/Cargo.toml

# The TUI is a separate workspace, so these provide convenient shortcuts
build-tui:
	cargo build --manifest-path tui/Cargo.toml

run-tui:
	cargo run --manifest-path tui/Cargo.toml
