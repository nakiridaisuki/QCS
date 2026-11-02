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
single: $(SINGLE_BINS)	# empty recipe: don't say "Nothing to be done" when up to date
	@:
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
             B.0.1 B.0.2 B.0.3 C.0.1 C.0.2 C.0.3 C.0.4 \
             final

# Every main-text table is fed by one or two suites; a suite runs only when
# its stamp is missing or its inputs changed.  Appendix tables run their own
# script on every invocation.
#
#   suite_ours.sh       u/cx baseline+local + RMSE check (seconds)
#   suite_cudaq.sh      CUDA-Q wall times + tensornet (JIT-dominated)
#   suite_nsys.sh       nsys traces; ours native + CUDA-Q on/off
#   suite_bench_gate.sh bench_gate for n=20/23/25 x fp32/fp64
STAMP_DIR    := single_gate/results/raw
OURS_STAMP   := $(STAMP_DIR)/.suite_ours.done
CUDAQ_STAMP  := $(STAMP_DIR)/.suite_cudaq.done
NSYS_STAMP   := $(STAMP_DIR)/.suite_nsys.done
NSYS_TRACE_STAMP := $(STAMP_DIR)/.suite_nsys_trace.done
BENCH_STAMP  := $(STAMP_DIR)/.suite_bench_gate.done
TABLE_STAMP_3.0.1 := $(OURS_STAMP)
TABLE_STAMP_4.1.1 := $(CUDAQ_STAMP)
TABLE_STAMP_4.2.1 := $(CUDAQ_STAMP)
TABLE_STAMP_4.3.1 := $(CUDAQ_STAMP)
TABLE_STAMP_4.3.2 := $(NSYS_STAMP)
TABLE_STAMP_4.3.3 := $(NSYS_STAMP)
TABLE_STAMP_6.2.1 := $(OURS_STAMP) $(NSYS_STAMP)
TABLE_STAMP_7.0.1 := $(NSYS_STAMP) $(CUDAQ_STAMP)
TABLE_STAMP_7.1.1 := $(BENCH_STAMP)
TABLE_STAMP_7.2.1 := $(NSYS_STAMP)
TABLE_STAMP_7.2.2 := $(NSYS_STAMP)
TABLE_STAMP_7.2.3 := $(BENCH_STAMP)
TABLE_STAMP_7.3.1 := $(BENCH_STAMP)
TABLE_STAMP_7.3.2 := $(BENCH_STAMP)
TABLE_STAMP_final := $(OURS_STAMP) $(NSYS_STAMP)
# appendix tables: one stamp each, re-measured only when the script or the
# binaries change
TABLE_STAMP_B.0.1 := $(STAMP_DIR)/.appB_swap.done
TABLE_STAMP_B.0.2 := $(STAMP_DIR)/.appB_swap.done
TABLE_STAMP_B.0.3 := $(STAMP_DIR)/.appB.0.3_swap_correctness.done
TABLE_STAMP_C.0.1 := $(STAMP_DIR)/.appC_width.done
TABLE_STAMP_C.0.2 := $(STAMP_DIR)/.appC.0.2_ncu_counters.done
TABLE_STAMP_C.0.3 := $(STAMP_DIR)/.appC_width.done
TABLE_STAMP_C.0.4 := $(STAMP_DIR)/.appC.0.4_end_to_end.done
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
TABLE_PARSED_final := 1
# appendix tables are printed from their CSVs by parse_data.py
TABLE_PARSED_B.0.1 := 1
TABLE_PARSED_B.0.2 := 1
TABLE_PARSED_B.0.3 := 1
TABLE_PARSED_C.0.1 := 1
TABLE_PARSED_C.0.2 := 1
TABLE_PARSED_C.0.3 := 1
TABLE_PARSED_C.0.4 := 1

# Freshness inputs visible from the host (binaries and generated circuits are
# container-side and intentionally not listed here).
OURS_DEPS   := single_gate/jobs/suite_ours.sh $(SINGLE_SRC)/local_gate.cu \
               $(SINGLE_SRC)/cuquantum_baseline.cu
CUDAQ_DEPS  := single_gate/jobs/suite_cudaq.sh \
               benchmarks/qft_n20.qasm benchmarks/qft_n23.qasm benchmarks/qft_n25.qasm
TRACE_DEPS  := single_gate/jobs/suite_nsys_trace.sh $(SINGLE_SRC)/local_gate.cu
REDUCE_DEPS := single_gate/jobs/suite_nsys_reduce.sh
BENCH_DEPS  := single_gate/jobs/suite_bench_gate.sh $(SINGLE_SRC)/bench_gate.cu

# Extra source deps for the shared appendix stamps (both branches).
$(STAMP_DIR)/.appB_swap.done: single_gate/scripts/legacy/appB/bench_gate.cu
$(STAMP_DIR)/.appC_width.done: single_gate/scripts/legacy/appC/bench_gate.cu \
                               single_gate/scripts/bench_prod.cu

ifeq ($(SG_DIRECT),1)
# ---- inside the container: stamps are produced by running the suites --------
$(OURS_STAMP): $(OURS_DEPS) $(SINGLE_BINS) $(UCX_QFT)
	bash single_gate/jobs/suite_ours.sh
	@touch $@

$(CUDAQ_STAMP): $(CUDAQ_DEPS)
	bash single_gate/jobs/suite_cudaq.sh
	@touch $@

$(NSYS_TRACE_STAMP): $(TRACE_DEPS) $(SINGLE_OUT)/single_local_gate $(NATIVE_QFT)
	bash single_gate/jobs/suite_nsys_trace.sh
	@touch $@

$(NSYS_STAMP): $(REDUCE_DEPS) $(NSYS_TRACE_STAMP)
	bash single_gate/jobs/suite_nsys_reduce.sh
	@touch $@

$(BENCH_STAMP): $(BENCH_DEPS) $(SINGLE_OUT)/single_bench_gate
	bash single_gate/jobs/suite_bench_gate.sh
	@touch $@

# appendix tables: one stamp each (B.0.1+B.0.2 share appB_swap.sh,
# C.0.1+C.0.3 share appC_width.sh -- one build, one set of runs, two CSVs)
$(STAMP_DIR)/.app%.done: single_gate/jobs/app%.sh $(SINGLE_BINS)
	bash $<
	@touch $@

$(STAMP_DIR)/.appC.0.4_end_to_end.done: single_gate/scripts/local_gate.cu $(NATIVE_QFT)

define TABLE_RULE
$(1): $$(TABLE_STAMP_$(1))
	@if [ -n "$$(TABLE_PARSED_$(1))" ]; then \
		$$(PYTHON) single_gate/scripts/parse_data.py -t $(1); \
	else \
		cat single_gate/results/tables/$$(TABLE_SCRIPT_$(1)).csv; \
	fi
endef
else ifeq ($(SLURM),1)
# ---- host + SLURM: one queued job per stamp (the job builds the stamp inside
# the container via slurm_container.sh).  -W so the tables print fresh data.
SLURM_PRODUCE = sbatch $(SBATCH_ARGS) -W --parsable single_gate/jobs/slurm_container.sh

$(OURS_STAMP): $(OURS_DEPS)
	@$(SLURM_PRODUCE) $@
$(CUDAQ_STAMP): $(CUDAQ_DEPS)
	@$(SLURM_PRODUCE) $@
$(BENCH_STAMP): $(BENCH_DEPS)
	@$(SLURM_PRODUCE) $@
$(NSYS_TRACE_STAMP): $(TRACE_DEPS)
	@$(SLURM_PRODUCE) $@
# the reduce phase is CPU-only: run it right here on the login node
$(NSYS_STAMP): $(REDUCE_DEPS) $(NSYS_TRACE_STAMP)
	@apptainer exec --pwd "$(CURDIR)" --bind "$(CURDIR)":"$(CURDIR)" "$(SIF)" \
	    env SG_DIRECT=1 make --no-print-directory $@
$(STAMP_DIR)/.app%.done: single_gate/jobs/app%.sh
	@$(SLURM_PRODUCE) $@

define TABLE_RULE
$(1): $$(TABLE_STAMP_$(1))
	@if [ -n "$$(TABLE_PARSED_$(1))" ]; then \
		python3 single_gate/scripts/parse_data.py -t $(1); \
	else \
		cat single_gate/results/tables/$$(TABLE_SCRIPT_$(1)).csv; \
	fi
endef
else
# ---- host direct: stamps are produced inside docker/apptainer ---------------
$(OURS_STAMP): $(OURS_DEPS)
	@single_gate/scripts/dispatch.sh stamp $@
$(CUDAQ_STAMP): $(CUDAQ_DEPS)
	@single_gate/scripts/dispatch.sh stamp $@
$(BENCH_STAMP): $(BENCH_DEPS)
	@single_gate/scripts/dispatch.sh stamp $@
$(NSYS_TRACE_STAMP): $(TRACE_DEPS)
	@single_gate/scripts/dispatch.sh stamp $@
$(NSYS_STAMP): $(REDUCE_DEPS) $(NSYS_TRACE_STAMP)
	@single_gate/scripts/dispatch.sh stamp $@
$(STAMP_DIR)/.app%.done: single_gate/jobs/app%.sh
	@single_gate/scripts/dispatch.sh stamp $@

define TABLE_RULE
$(1): $$(TABLE_STAMP_$(1))
	@if [ -n "$$(TABLE_PARSED_$(1))" ]; then \
		python3 single_gate/scripts/parse_data.py -t $(1); \
	else \
		cat single_gate/results/tables/$$(TABLE_SCRIPT_$(1)).csv; \
	fi
endef
endif
$(foreach id,$(TABLE_IDS),$(eval $(call TABLE_RULE,$(id))))

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



