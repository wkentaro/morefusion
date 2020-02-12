all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install_anaconda3:
	@Makefile.scripts/install_anaconda3.sh

install_v-hacd: install_anaconda3
	@Makefile.scripts/install_v-hacd.sh

install_binvox: install_anaconda3
	@Makefile.scripts/install_binvox.sh

install: install_anaconda3 install_v-hacd install_binvox
	@Makefile.scripts/install.sh

lint: install_anaconda3
	@Makefile.scripts/lint.sh

test: install_anaconda3
	@Makefile.scripts/test.sh

check: install_anaconda3
	@Makefile.scripts/check.sh
