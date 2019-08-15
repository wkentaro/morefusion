all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install_anaconda3:
	@.make/install_anaconda3.sh

install_v-hacd: install_anaconda3
	@.make/install_v-hacd.sh

install_binvox: install_anaconda3
	@.make/install_binvox.sh

install_pykfusion: install_anaconda3
	@.make/install_pykfusion.sh

install: install_anaconda3 install_v-hacd install_binvox install_pykfusion
	@.make/install.sh

lint:
	@.make/lint.sh

test:
	@.make/test.sh

check:
	@.make/check.sh
