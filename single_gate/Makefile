# Single-gate binaries for SINGLE_GATE_REPORT.
#
# This file is included by the repo-root Makefile, so it must be invoked from
# the repo root (`make single`), not with `make -C single_gate`.
# NVCC/NVCCFLAGS/SVLIB come from the including Makefile.

SINGLE_SRC  ?= single_gate/scripts
SINGLE_OUT  ?= single_gate/build
SINGLE_BINS := $(SINGLE_OUT)/single_baseline $(SINGLE_OUT)/single_local_gate \
               $(SINGLE_OUT)/single_bench_gate

# Host-side dispatch.  SG_DIRECT=1 is only set inside the container.
IMAGE  ?= single-gate:cudaq0.16
SIF    := $(CURDIR)/single_gate/env/single-gate-cudaq0.16.sif
# Optional: only set these when the default Apptainer cache/tmp is too small.
# Empty means use Apptainer's normal defaults (~/.apptainer, /tmp).
APPTAINER_TMPDIR   ?=
APPTAINER_CACHEDIR ?=
RUNNER ?= auto
export IMAGE RUNNER

.PHONY: single single-docker single-apptainer
ifeq ($(SLURM),1)
single:
	@:
else ifeq ($(SG_DIRECT),1)
single: $(SINGLE_BINS)
else
single:
	@single_gate/scripts/dispatch.sh build
endif

$(SINGLE_OUT):
	mkdir -p $(SINGLE_OUT)

# Separate binaries on purpose: they must not drag the current pipeline into a
# comparison, and their names must not collide with the live bench_gate.
$(SINGLE_OUT)/single_baseline: $(SINGLE_SRC)/cuquantum_baseline.cu | $(SINGLE_OUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SVLIB)

$(SINGLE_OUT)/single_local_gate: $(SINGLE_SRC)/local_gate.cu | $(SINGLE_OUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SINGLE_OUT)/single_bench_gate: $(SINGLE_SRC)/bench_gate.cu | $(SINGLE_OUT)
	$(NVCC) $(NVCCFLAGS) -I$(SINGLE_SRC) -o $@ $< $(SVLIB)


# ---------------------------------------------------------------- container images
single-docker:
	docker build -f single_gate/env/Dockerfile -t $(IMAGE) .

single-apptainer:
	@command -v docker >/dev/null || { echo "!! docker not found"; exit 1; }
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || { \
		echo "!! Docker image $(IMAGE) not found; run: make single-docker"; exit 1; }
	tar=$(CURDIR)/single-gate-cudaq0.16.tar; \
	tmp='$(APPTAINER_TMPDIR)'; cache='$(APPTAINER_CACHEDIR)'; \
	trap 'rm -f "$$tar"' EXIT; \
	docker save $(IMAGE) -o "$$tar"; \
	if [ -n "$$tmp" ]; then mkdir -p "$$tmp"; export TMPDIR="$$tmp" APPTAINER_TMPDIR="$$tmp"; fi; \
	if [ -n "$$cache" ]; then mkdir -p "$$cache"; export APPTAINER_CACHEDIR="$$cache"; fi; \
	apptainer build --force "$(SIF)" "docker-archive:$$tar"

# ---------------------------------------------------------- IR circuits
# Tables consume these; regenerate from benchmarks/*.qasm when missing or stale,
# so a fresh clone can run the tables directly. Only wired up inside the
# container (SG_DIRECT=1): the host has no qiskit, and the container re-invokes
# make for the table target anyway.
UCX_3      := $(addprefix $(SINGLE_SRC)/circuits_ucx/,vqe_n16.circuit qft_n20.circuit qft_n23.circuit qft_n25.circuit)
UCX_QFT    := $(addprefix $(SINGLE_SRC)/circuits_ucx/,qft_n20.circuit qft_n23.circuit qft_n25.circuit)
NATIVE_QFT := $(addprefix $(SINGLE_SRC)/circuits_native/,qft_n20.circuit qft_n23.circuit qft_n25.circuit)

$(SINGLE_SRC)/circuits_ucx/%.circuit: benchmarks/%.qasm
	$(PYTHON) $(SINGLE_SRC)/qiskit_qasm_parser.py $< $@

$(SINGLE_SRC)/circuits_native/%.circuit: benchmarks/%.qasm
	$(PYTHON) $(SINGLE_SRC)/qiskit_qasm_parser.py --native $< $@

# ---------------------------------------------------------- run one report table
# make single 3.0.1   -> that table
# make single 7.3     -> 7.3.1 + 7.3.2
# make single 7       -> every chapter-7 table
# make single B       -> appendix B
TABLE_IDS := 3.0.1 4.1.1 4.2.1 4.3.1 4.3.2 4.3.3 6.2.1 \
             7.0.1 7.1.1 7.2.1 7.2.2 7.2.3 7.3.1 7.3.2 \
             B.0.1 B.0.2 B.0.3 C.0.1 C.0.2 C.0.3 C.0.4

TABLE_SCRIPT_3.0.1 := sec3.0.1_whole_circuit
TABLE_SCRIPT_4.1.1 := sec4.1.1_cudaq_statevector
TABLE_SCRIPT_4.2.1 := sec4.2.1_cudaq_tensornet
TABLE_SCRIPT_4.3.1 := sec4.3.1_cudaq_fusion
TABLE_SCRIPT_4.3.2 := sec4.3.2_cudaq_kernel_sum
TABLE_SCRIPT_4.3.3 := sec4.3.3_cudaq_kernels
TABLE_SCRIPT_6.2.1 := sec6.2.1_native_speedup
TABLE_SCRIPT_7.0.1 := sec7.0.1_same_decomposition
TABLE_SCRIPT_7.1.1 := sec7.1.1_d2h
TABLE_SCRIPT_7.2.1 := sec7.2.1_kernel_sums
TABLE_SCRIPT_7.2.2 := sec7.2.2_per_operation
TABLE_SCRIPT_7.2.3 := sec7.2.3_per_gate
TABLE_SCRIPT_7.3.1 := sec7.3.1_u_target_sweep
TABLE_SCRIPT_7.3.2 := sec7.3.2_bit_positions
TABLE_SCRIPT_B.0.1 := appB.0.1_swap_bandwidth
TABLE_SCRIPT_B.0.2 := appB.0.2_swap_before_after
TABLE_SCRIPT_B.0.3 := appB.0.3_swap_correctness
TABLE_SCRIPT_C.0.1 := appC.0.1_u_target_before
TABLE_SCRIPT_C.0.2 := appC.0.2_ncu_counters
TABLE_SCRIPT_C.0.3 := appC.0.3_wide_shfl
TABLE_SCRIPT_C.0.4 := appC.0.4_end_to_end

# tables parse_data.py knows how to print; appendix CSVs are printed directly
TABLE_PARSED_3.0.1 := 1
TABLE_PARSED_4.1.1 := 1
TABLE_PARSED_4.2.1 := 1
TABLE_PARSED_4.3.1 := 1
TABLE_PARSED_4.3.2 := 1
TABLE_PARSED_4.3.3 := 1
TABLE_PARSED_6.2.1 := 1
TABLE_PARSED_7.0.1 := 1
TABLE_PARSED_7.1.1 := 1
TABLE_PARSED_7.2.1 := 1
TABLE_PARSED_7.2.2 := 1
TABLE_PARSED_7.2.3 := 1
TABLE_PARSED_7.3.1 := 1
TABLE_PARSED_7.3.2 := 1

ifeq ($(SG_DIRECT),1)
define TABLE_RULE
$(1):
	bash single_gate/jobs/$$(TABLE_SCRIPT_$(1)).sh
	@if [ -n "$$(TABLE_PARSED_$(1))" ]; then \
		$$(PYTHON) single_gate/scripts/parse_data.py -t $(1); \
	fi
endef
else
define TABLE_RULE
$(1):
	@single_gate/scripts/dispatch.sh run $(1)
endef
endif
$(foreach id,$(TABLE_IDS),$(eval $(call TABLE_RULE,$(id))))

# Tables that consume IR circuits get them as prerequisites (rules above).
ifeq ($(SG_DIRECT),1)
3.0.1: $(UCX_3)
6.2.1: $(UCX_QFT) $(NATIVE_QFT)
7.0.1 7.2.1 7.2.2 C.0.4: $(NATIVE_QFT)
endif

define PARSE_RULE
parse-$(1):
	@if [ -n "$$(TABLE_PARSED_$(1))" ]; then \
		$$(PYTHON) single_gate/scripts/parse_data.py -t $(1); \
	else \
		cat single_gate/results/tables/$$(TABLE_SCRIPT_$(1)).csv; \
	fi
endef
$(foreach id,$(TABLE_IDS),$(eval $(call PARSE_RULE,$(id))))

# X.Y and X aliases
3: 3.0.1
3.0: 3.0.1
4: 4.1.1 4.2.1 4.3.1 4.3.2 4.3.3
4.1: 4.1.1
4.2: 4.2.1
4.3: 4.3.1 4.3.2 4.3.3
6: 6.2.1
6.2: 6.2.1
7: 7.0.1 7.1.1 7.2.1 7.2.2 7.2.3 7.3.1 7.3.2
7.0: 7.0.1
7.1: 7.1.1
7.2: 7.2.1 7.2.2 7.2.3
7.3: 7.3.1 7.3.2
B: B.0.1 B.0.2 B.0.3
B.0: B.0.1 B.0.2 B.0.3
C: C.0.1 C.0.2 C.0.3 C.0.4
C.0: C.0.1 C.0.2 C.0.3 C.0.4

.PHONY: $(TABLE_IDS) 3 3.0 4 4.1 4.2 4.3 6 6.2 \
        7 7.0 7.1 7.2 7.3 B B.0 C C.0


# ID -> table list, used by single_parse (same prefixes as above)
TABLE_LIST_3 := 3.0.1
TABLE_LIST_3.0 := 3.0.1
TABLE_LIST_4 := 4.1.1 4.2.1 4.3.1 4.3.2 4.3.3
TABLE_LIST_4.1 := 4.1.1
TABLE_LIST_4.2 := 4.2.1
TABLE_LIST_4.3 := 4.3.1 4.3.2 4.3.3
TABLE_LIST_6 := 6.2.1
TABLE_LIST_6.2 := 6.2.1
TABLE_LIST_7 := 7.0.1 7.1.1 7.2.1 7.2.2 7.2.3 7.3.1 7.3.2
TABLE_LIST_7.0 := 7.0.1
TABLE_LIST_7.1 := 7.1.1
TABLE_LIST_7.2 := 7.2.1 7.2.2 7.2.3
TABLE_LIST_7.3 := 7.3.1 7.3.2
TABLE_LIST_B := B.0.1 B.0.2 B.0.3
TABLE_LIST_B.0 := B.0.1 B.0.2 B.0.3
TABLE_LIST_C := C.0.1 C.0.2 C.0.3 C.0.4
TABLE_LIST_C.0 := C.0.1 C.0.2 C.0.3 C.0.4
TABLE_LIST_all := $(TABLE_IDS)
# single table ids are also valid singe_parse IDs
$(foreach id,$(TABLE_IDS),$(eval TABLE_LIST_$(id) := $(id)))

.PHONY: single_parse $(addprefix parse-,$(TABLE_IDS))
ifeq ($(SG_DIRECT),1)
single_parse:
	@if [ -z "$(strip $(ID))" ]; then \
		echo "usage: make single_parse ID=<X.Y.Z|X.Y|X|all>"; exit 1; fi
	@if [ -z "$(TABLE_LIST_$(ID))" ]; then \
		echo "unknown table id: $(ID)"; exit 2; fi
	@$(MAKE) --no-print-directory $(addprefix parse-,$(TABLE_LIST_$(ID)))
else
single_parse:
	@single_gate/scripts/dispatch.sh parse "$(ID)"
endif
