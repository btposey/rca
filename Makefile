.PHONY: check run

check:
	bash scripts/check_deps.sh

run:
	bash scripts/deploy_native_vllm.sh
